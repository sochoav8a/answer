"""
train.py

Script principal mejorado para entrenar el modelo CycleGAN con SwinIR:
- Validación periódica durante el entrenamiento
- Monitorización detallada con TensorBoard
- Visualización de resultados intermedios
- Checkpoint automático y criterios de early stopping
- Manejo de errores y reanudación de entrenamiento
- Optimización de recursos computacionales
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
import logging
import sys
from datetime import datetime

import config
from data.dataset import ConfocalDataset, get_default_transforms, get_validation_transforms
from models.cyclegan_model import CycleGANModel
from utils.metrics import calculate_psnr, calculate_ssim

# Configuración de logging
def setup_logging(log_dir='logs'):
    """Configura el sistema de logging para el entrenamiento."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configurar logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, epoch, optimizer_G=None, optimizer_D=None, loss_G=None, loss_D=None, config=None, best=False):
    """
    Guarda un checkpoint completo del modelo y optimizadores.
    
    Parámetros:
        model (CycleGANModel): El modelo a guardar
        epoch (int): Época actual
        optimizer_G: Optimizador de generadores (opcional)
        optimizer_D: Optimizador de discriminadores (opcional)
        loss_G (float): Pérdida actual del generador (opcional)
        loss_D (float): Pérdida actual del discriminador (opcional)
        config: Configuración global
        best (bool): Si True, guarda como mejor modelo
    """
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)
    
    # Determinar nombre del checkpoint
    if best:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_best.pth")
    else:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_epoch_{epoch}.pth")
    
    # Preparar estado del modelo y optimizadores
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': model.netG.state_dict(),
        'netF_state_dict': model.netF.state_dict(),
        'netD_HR_state_dict': model.netD_HR.state_dict(),
        'netD_LR_state_dict': model.netD_LR.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
        'scaler_G_state_dict': model.scaler_G.state_dict(),
        'scaler_D_state_dict': model.scaler_D.state_dict(),
    }
    
    # Guardar métricas si están disponibles
    if loss_G is not None:
        checkpoint['loss_G'] = loss_G
    if loss_D is not None:
        checkpoint['loss_D'] = loss_D
    
    # Guardar el checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    if best:
        logging.info(f"✅ Mejor modelo guardado en: {checkpoint_path}")
    else:
        logging.info(f"📦 Checkpoint guardado en: {checkpoint_path}")

def load_checkpoint(model, checkpoint_path, optimizer_G=None, optimizer_D=None):
    """
    Carga un checkpoint existente para continuar el entrenamiento.
    
    Parámetros:
        model (CycleGANModel): El modelo donde cargar los pesos
        checkpoint_path (str): Ruta al archivo de checkpoint
        optimizer_G: Optimizador de generadores (opcional)
        optimizer_D: Optimizador de discriminadores (opcional)
        
    Retorna:
        int: Número de época desde donde continuar
        float: Mejor pérdida del generador (o None si no disponible)
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"❌ No se encontró checkpoint en: {checkpoint_path}")
        return 0, None
    
    logging.info(f"📂 Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Cargar estados del modelo
    model.netG.load_state_dict(checkpoint['netG_state_dict'])
    model.netF.load_state_dict(checkpoint['netF_state_dict'])
    model.netD_HR.load_state_dict(checkpoint['netD_HR_state_dict'])
    model.netD_LR.load_state_dict(checkpoint['netD_LR_state_dict'])
    
    # Cargar estados de optimizadores si están disponibles y proporcionados
    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D is not None and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
    # Cargar escaladores
    if 'scaler_G_state_dict' in checkpoint:
        model.scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
    if 'scaler_D_state_dict' in checkpoint:
        model.scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
    
    # Obtener época y pérdida del generador
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss_G = checkpoint.get('loss_G', float('inf'))
    
    logging.info(f"✅ Checkpoint cargado: continuar desde época {start_epoch}")
    
    return start_epoch, best_loss_G

def visualize_results(model, fixed_samples, writer, epoch, device):
    """
    Genera visualizaciones para TensorBoard mostrando los resultados del modelo.
    
    Parámetros:
        model (CycleGANModel): Modelo CycleGAN entrenado
        fixed_samples (dict): Muestras fijas para visualización constante
        writer (SummaryWriter): Writer de TensorBoard
        epoch (int): Época actual
        device: Dispositivo de procesamiento
    """
    model.eval()
    
    with torch.no_grad():
        # Mover las muestras fijas al dispositivo
        lr_samples = fixed_samples['LR'].to(device)
        hr_samples = fixed_samples['HR'].to(device)
        
        # Generar transformaciones en ambas direcciones
        fake_hr = model.netG(lr_samples)
        rec_lr = model.netF(fake_hr)
        
        fake_lr = model.netF(hr_samples)
        rec_hr = model.netG(fake_lr)
        
        # Desnormalizar para visualización (-1,1 -> 0,1)
        def denorm(x):
            return (x + 1) / 2
        
        # Crear grids para cada dirección de transformación
        # LR -> HR
        lr_to_hr_grid = vutils.make_grid(
            torch.cat([
                denorm(lr_samples),
                denorm(fake_hr),
                denorm(rec_lr)
            ], dim=0),
            nrow=lr_samples.size(0),
            normalize=False,
            padding=2
        )
        
        # HR -> LR
        hr_to_lr_grid = vutils.make_grid(
            torch.cat([
                denorm(hr_samples),
                denorm(fake_lr),
                denorm(rec_hr)
            ], dim=0),
            nrow=hr_samples.size(0),
            normalize=False,
            padding=2
        )
        
        # Escribir imágenes en TensorBoard
        writer.add_image(f'LR→HR (LR/Fake-HR/Rec-LR)', lr_to_hr_grid, epoch)
        writer.add_image(f'HR→LR (HR/Fake-LR/Rec-HR)', hr_to_lr_grid, epoch)
    
    model.train()

def validate(model, val_dataloader, device, writer=None, epoch=None):
    """
    Ejecuta una validación completa del modelo y calcula métricas.
    
    Parámetros:
        model (CycleGANModel): Modelo a validar
        val_dataloader: DataLoader con datos de validación
        device: Dispositivo de procesamiento
        writer (SummaryWriter): Writer de TensorBoard (opcional)
        epoch (int): Época actual para logging (opcional)
        
    Retorna:
        dict: Diccionario con métricas de validación
    """
    model.eval()
    
    # Métricas a seguir
    psnr_values = []
    ssim_values = []
    val_loss_G = 0.0
    val_loss_D = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc="Validación", leave=False)):
            # Procesar batch
            model.set_input(data)
            model.forward()
            
            # Obtener tensores relevantes y desnormalizar para métricas
            fake_hr = (model.fake_HR.detach() + 1) / 2  # [-1,1] -> [0,1]
            real_hr = (model.real_HR.detach() + 1) / 2
            
            # Calcular métricas cuantitativas
            batch_psnr = calculate_psnr(fake_hr, real_hr)
            batch_ssim = calculate_ssim(fake_hr, real_hr)
            
            psnr_values.append(batch_psnr)
            ssim_values.append(batch_ssim)
            
            # Calcular pérdidas (sin retropropagación)
            model.backward_G()
            model.backward_D()
            
            val_loss_G += model.loss_G.item()
            val_loss_D += model.loss_D.item()
            
            num_samples += 1
    
    # Promediar métricas
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    avg_loss_G = val_loss_G / num_samples if num_samples > 0 else 0
    avg_loss_D = val_loss_D / num_samples if num_samples > 0 else 0
    
    # Escribir en TensorBoard si está disponible
    if writer and epoch is not None:
        writer.add_scalar('Validation/PSNR', avg_psnr, epoch)
        writer.add_scalar('Validation/SSIM', avg_ssim, epoch)
        writer.add_scalar('Validation/Loss_G', avg_loss_G, epoch)
        writer.add_scalar('Validation/Loss_D', avg_loss_D, epoch)
    
    # Reactivar modo entrenamiento
    model.train()
    
    # Crear diccionario de resultados
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss_G': avg_loss_G,
        'loss_D': avg_loss_D
    }
    
    logging.info(f"📊 Validación: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}, Loss_G = {avg_loss_G:.4f}")
    
    return metrics

def init_weights(m):
    """Inicializa pesos con distribución normal."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train(args):
    """
    Función principal de entrenamiento con todas las mejoras.
    
    Parámetros:
        args: Argumentos de línea de comandos
    """
    # 1. Configuración inicial y semilla para reproducibilidad
    config.setup_seed(config.RANDOM_SEED)
    logger = setup_logging()
    
    device = config.DEVICE
    logging.info(f"🚀 Iniciando entrenamiento en: {device}")
    logging.info(f"📋 Parámetros: Épocas={config.EPOCHS}, Batch={config.BATCH_SIZE}, LR_G={config.LR_GENERATOR}")
    
    # 2. Inicializar TensorBoard y directorios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.MODEL_NAME}_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    
    # 3. Preparar datasets y dataloaders
    # Transformaciones para entrenamiento (con aumentación)
    train_transform = get_default_transforms(config.IMG_SIZE)
    
    # Transformaciones para validación (sin aumentación)
    val_transform = get_validation_transforms(config.IMG_SIZE)
    
    # Dataset de entrenamiento
    train_dataset = ConfocalDataset(
        lr_dir=config.DATA_PATH,
        transform_lr=train_transform,
        transform_hr=train_transform,
        paired=False,
        part_data=config.PART_DATA
    )
    
    # Dataset de validación (pareado)
    val_dataset = None
    if os.path.exists(config.VALID_PATH):
        val_dataset = ConfocalDataset(
            lr_dir=config.VALID_PATH,
            transform_lr=val_transform,
            transform_hr=val_transform,
            paired=True,
            part_data=1.0  # Usar todos los datos de validación
        )
        logging.info(f"✅ Dataset de validación cargado: {len(val_dataset)} muestras")
    
    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS, 
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    
    # 4. Inicializar el modelo CycleGAN
    model = CycleGANModel(config)
    model.to(device)
    
    # Aplicar inicialización de pesos personalizada
    # model.apply(init_weights)
    
    # 5. Preparar para continuar entrenamiento si se especifica
    start_epoch = 0
    best_loss_G = float('inf')
    
    if args.resume:
        resume_path = args.resume
        if resume_path.lower() == 'auto':
            # Buscar último checkpoint automáticamente
            checkpoints = [f for f in os.listdir(config.CHECKPOINTS_DIR) 
                         if f.startswith(config.MODEL_NAME) and f.endswith('.pth')]
            if checkpoints:
                epochs = [int(f.split('_epoch_')[-1].split('.pth')[0]) 
                       for f in checkpoints if '_epoch_' in f]
                if epochs:
                    max_epoch = max(epochs)
                    resume_path = os.path.join(
                        config.CHECKPOINTS_DIR, 
                        f"{config.MODEL_NAME}_epoch_{max_epoch}.pth"
                    )
                else:
                    resume_path = os.path.join(
                        config.CHECKPOINTS_DIR,
                        f"{config.MODEL_NAME}_best.pth"
                    )
        
        if os.path.exists(resume_path):
            start_epoch, best_loss_G = load_checkpoint(
                model, 
                resume_path,
                model.optimizer_G,
                model.optimizer_D
            )
    
    # 6. Preparar muestras fijas para visualización consistente
    fixed_samples = next(iter(train_dataloader))
    fixed_samples = {
        'LR': fixed_samples['LR'][:4].to(device),  # Limitar a 4 muestras
        'HR': fixed_samples['HR'][:4].to(device)
    }
    
    # 7. Entrenamiento principal
    total_epochs = config.EPOCHS
    logging.info(f"🏃 Comenzando entrenamiento para {total_epochs} épocas")
    
    # Parámetros para early stopping
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        
        # Inicializar contadores para esta época
        running_loss_G = 0.0
        running_loss_D = 0.0
        count = 0
        
        # Barra de progreso para esta época
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Época {epoch+1}/{total_epochs}", 
            leave=True,
            unit="batch"
        )
        
        # Iterar sobre todos los batches
        for i, data in enumerate(progress_bar):
            count += 1
            
            # Procesar batch
            model.set_input(data)
            model.optimize_parameters()
            
            # Obtener y acumular pérdidas
            losses = model.get_current_losses()
            running_loss_G += losses['loss_G']
            running_loss_D += losses['loss_D']
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss_G': f"{losses['loss_G']:.4f}",
                'loss_D': f"{losses['loss_D']:.4f}"
            })
            
            # Logging periódico a TensorBoard (cada N batches)
            if (i + 1) % config.PRINT_FREQ == 0:
                step = epoch * len(train_dataloader) + i
                for key, value in losses.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'Batch/{key}', value, step)
        
        # Calcular métricas promedio de la época
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = running_loss_G / count
        avg_loss_D = running_loss_D / count
        
        # Actualizar la tasa de aprendizaje
        model.update_learning_rate()
        
        # Logging de época
        logging.info(f"⏱️ Época [{epoch+1}/{total_epochs}] completada en {epoch_time:.2f}s")
        logging.info(f"📉 Pérdidas - G: {avg_loss_G:.4f}, D: {avg_loss_D:.4f}")
        
        # Escribir métricas de época en TensorBoard
        writer.add_scalar("Epoch/Generator_Loss", avg_loss_G, epoch)
        writer.add_scalar("Epoch/Discriminator_Loss", avg_loss_D, epoch)
        writer.add_scalar("Epoch/Learning_Rate_G", model.optimizer_G.param_groups[0]['lr'], epoch)
        writer.add_scalar("Epoch/Learning_Rate_D", model.optimizer_D.param_groups[0]['lr'], epoch)
        writer.add_scalar("Epoch/Time", epoch_time, epoch)
        
        # Visualizar resultados actuales
        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_results(model, fixed_samples, writer, epoch, device)
        
        # Validación
        if val_dataloader and ((epoch + 1) % 5 == 0 or epoch == 0):
            val_metrics = validate(model, val_dataloader, device, writer, epoch)
        
        # Determinar si este es el mejor modelo hasta ahora
        is_best = avg_loss_G < best_loss_G
        if is_best:
            best_loss_G = avg_loss_G
            patience_counter = 0
            save_checkpoint(model, epoch, loss_G=avg_loss_G, loss_D=avg_loss_D, config=config, best=True)
        else:
            patience_counter += 1
        
        # Guardar checkpoint periódico
        if (epoch + 1) % config.SAVE_FREQ == 0:
            save_checkpoint(model, epoch, loss_G=avg_loss_G, loss_D=avg_loss_D, config=config)
        
        # Early stopping
        if patience > 0 and patience_counter >= patience:
            logging.info(f"🛑 Early stopping tras {patience} épocas sin mejora")
            break
    
    # 8. Finalización
    writer.close()
    logging.info("✅ Entrenamiento completado")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de CycleGAN para imágenes confocales')
    parser.add_argument('--resume', type=str, default='', help='Ruta al checkpoint para continuar entrenamiento, o "auto" para detectar automáticamente')
    parser.add_argument('--patience', type=int, default=20, help='Paciencia para early stopping (0 para desactivar)')
    args = parser.parse_args()
    
    # Iniciar entrenamiento
    train(args)