"""
train_ddp.py

Versión mejorada del script de entrenamiento para multi-GPU usando DistributedDataParallel (DDP):
- Inicialización robusta del entorno distribuido
- Optimizaciones de comunicación entre procesos
- División eficiente de datos entre múltiples GPUs
- Sincronización de métricas y resultados
- Todas las mejoras de monitorización y optimización de train.py
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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
def setup_logging(rank, log_dir='logs'):
    # Configuración básica
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_rank{rank}_{timestamp}.log')
    
    # Configuración básica sin el rango en el formato
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formato estándar sin rango incluido
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, epoch, config, best=False):
    """
    Guarda un checkpoint del modelo entrenado con DDP.
    Solo el proceso principal (rank=0) realiza el guardado.
    
    Parámetros:
        model: Modelo DDP envuelto
        epoch: Época actual
        config: Configuración global
        best: Si es el mejor modelo hasta ahora
    """
    # Solo el proceso principal (rank 0) guarda checkpoints
    if dist.get_rank() != 0:
        return
    
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)
    
    # Determinar nombre de archivo
    if best:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_best.pth")
    else:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_epoch_{epoch}.pth")
    
    # Obtener el módulo base sin el wrapper DDP
    model_module = model.module
    
    # Preparar estado completo del modelo
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': model_module.netG.state_dict(),
        'netF_state_dict': model_module.netF.state_dict(),
        'netD_HR_state_dict': model_module.netD_HR.state_dict(),
        'netD_LR_state_dict': model_module.netD_LR.state_dict(),
        'optimizer_G_state_dict': model_module.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model_module.optimizer_D.state_dict(),
        'scaler_G_state_dict': model_module.scaler_G.state_dict(),
        'scaler_D_state_dict': model_module.scaler_D.state_dict(),
    }
    
    # Guardar checkpoint
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"📦 Checkpoint guardado en: {checkpoint_path}")

def load_checkpoint(model, checkpoint_path):
    """
    Carga un checkpoint existente en modelo DDP.
    
    Parámetros:
        model: Modelo DDP envuelto
        checkpoint_path: Ruta al checkpoint
        
    Retorna:
        epoch: Época desde la que continuar
        best_loss: Mejor pérdida registrada
    """
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Cargar en CPU primero
    
    # Obtener el módulo base
    model_module = model.module
    
    # Cargar estados
    model_module.netG.load_state_dict(checkpoint['netG_state_dict'])
    model_module.netF.load_state_dict(checkpoint['netF_state_dict'])
    model_module.netD_HR.load_state_dict(checkpoint['netD_HR_state_dict'])
    model_module.netD_LR.load_state_dict(checkpoint['netD_LR_state_dict'])
    
    # Cargar optimizadores
    model_module.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    model_module.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # Cargar escaladores si existen
    if 'scaler_G_state_dict' in checkpoint:
        model_module.scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
    if 'scaler_D_state_dict' in checkpoint:
        model_module.scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
    
    # Obtener época y mejor pérdida
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss_G', float('inf'))
    
    return start_epoch, best_loss

def visualize_results(model, fixed_samples, writer, epoch, rank):
    """
    Genera visualizaciones para TensorBoard desde el proceso principal.
    
    Parámetros:
        model: Modelo DDP
        fixed_samples: Muestras fijas para visualización
        writer: Writer de TensorBoard
        epoch: Época actual
        rank: Rango del proceso actual
    """
    # Solo el proceso principal (rank 0) genera visualizaciones
    if rank != 0:
        return
    
    model.eval()
    
    with torch.no_grad():
        # Acceder al módulo base
        model_module = model.module
        
        # Procesar muestras fijas
        model_module.set_input(fixed_samples)
        model_module.forward()
        
        # Desnormalizar para visualización (-1,1 -> 0,1)
        def denorm(x):
            return (x + 1) / 2
        
        # Obtener imágenes para visualización
        real_lr = denorm(model_module.real_LR)
        fake_hr = denorm(model_module.fake_HR)
        rec_lr = denorm(model_module.rec_LR)
        
        real_hr = denorm(model_module.real_HR)
        fake_lr = denorm(model_module.fake_LR)
        rec_hr = denorm(model_module.rec_HR)
        
        # Crear grids para visualización
        lr_to_hr_grid = vutils.make_grid(
            torch.cat([real_lr, fake_hr, rec_lr], dim=0),
            nrow=real_lr.size(0),
            normalize=False,
            padding=2
        )
        
        hr_to_lr_grid = vutils.make_grid(
            torch.cat([real_hr, fake_lr, rec_hr], dim=0),
            nrow=real_hr.size(0),
            normalize=False,
            padding=2
        )
        
        # Escribir a TensorBoard
        writer.add_image('LR→HR (Original/Generado/Reconstruido)', lr_to_hr_grid, epoch)
        writer.add_image('HR→LR (Original/Generado/Reconstruido)', hr_to_lr_grid, epoch)
    
    model.train()

def validate_ddp(model, val_loader, device, rank, world_size, writer=None, epoch=None):
    """
    Ejecuta validación distribuida y sincroniza métricas entre procesos.
    
    Parámetros:
        model: Modelo DDP
        val_loader: Dataloader de validación con DistributedSampler
        device: Dispositivo para cálculos
        rank: Rango del proceso actual
        world_size: Número total de procesos
        writer: Writer de TensorBoard (solo para rank 0)
        epoch: Época actual
        
    Retorna:
        dict: Métricas de validación sincronizadas
    """
    model.eval()
    
    # Tensores para acumular métricas (en GPU para reducción eficiente)
    psnr_sum = torch.tensor(0.0, device=device)
    ssim_sum = torch.tensor(0.0, device=device)
    loss_g_sum = torch.tensor(0.0, device=device)
    loss_d_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device)
    
    with torch.no_grad():
        for data in val_loader:
            # Procesar batch
            model.module.set_input(data)
            model.module.forward()
            
            # Desnormalizar para métricas
            fake_hr = (model.module.fake_HR + 1) / 2
            real_hr = (model.module.real_HR + 1) / 2
            
            # Calcular métricas
            batch_psnr = calculate_psnr(fake_hr, real_hr)
            batch_ssim = calculate_ssim(fake_hr, real_hr)
            
            # Calcular pérdidas sin retropropagación
            model.module.backward_G()
            model.module.backward_D()
            
            # Acumular métricas
            psnr_sum += torch.tensor(batch_psnr, device=device)
            ssim_sum += torch.tensor(batch_ssim, device=device)
            loss_g_sum += model.module.loss_G.detach()
            loss_d_sum += model.module.loss_D.detach()
            count += 1
    
    # Sincronizar métricas entre todos los procesos
    dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_g_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_d_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    # Calcular promedios
    avg_psnr = (psnr_sum / count).item()
    avg_ssim = (ssim_sum / count).item()
    avg_loss_g = (loss_g_sum / count).item()
    avg_loss_d = (loss_d_sum / count).item()
    
    # Solo el proceso principal escribe en TensorBoard
    if rank == 0 and writer and epoch is not None:
        writer.add_scalar('Validation/PSNR', avg_psnr, epoch)
        writer.add_scalar('Validation/SSIM', avg_ssim, epoch)
        writer.add_scalar('Validation/Loss_G', avg_loss_g, epoch)
        writer.add_scalar('Validation/Loss_D', avg_loss_d, epoch)
        logging.info(f"📊 Validación: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, Loss_G={avg_loss_g:.4f}")
    
    model.train()
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss_G': avg_loss_g,
        'loss_D': avg_loss_d
    }

def train_epoch(epoch, model, train_loader, device, rank, world_size, writer=None, total_epochs=0):
    """
    Entrena una época con DDP y sincroniza métricas.
    
    Parámetros:
        epoch: Época actual
        model: Modelo DDP
        train_loader: DataLoader con DistributedSampler
        device: Dispositivo para cálculos
        rank: Rango del proceso actual
        world_size: Número total de procesos
        writer: Writer de TensorBoard (solo para rank 0)
        total_epochs: Número total de épocas
        
    Retorna:
        dict: Métricas promedio de la época
    """
    model.train()
    
    # Tensores para acumular pérdidas (en GPU para reducción eficiente)
    loss_g_sum = torch.tensor(0.0, device=device)
    loss_d_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device)
    
    # Crear barra de progreso solo en el proceso principal
    if rank == 0:
        progress_bar = tqdm(
            train_loader, 
            desc=f"Época {epoch+1}/{total_epochs}",
            leave=True
        )
    else:
        progress_bar = train_loader
    
    epoch_start = time.time()
    
    for data in progress_bar:
        # Procesar batch
        model.module.set_input(data)
        model.module.optimize_parameters()
        
        # Obtener pérdidas
        losses = model.module.get_current_losses()
        
        # Acumular pérdidas
        loss_g_sum += torch.tensor(losses['loss_G'], device=device)
        loss_d_sum += torch.tensor(losses['loss_D'], device=device)
        count += 1
        
        # Actualizar barra de progreso en proceso principal
        if rank == 0:
            progress_bar.set_postfix({
                'loss_G': f"{losses['loss_G']:.4f}",
                'loss_D': f"{losses['loss_D']:.4f}"
            })
    
    # Sincronizar métricas entre procesos
    dist.all_reduce(loss_g_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_d_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    # Calcular promedios
    avg_loss_g = (loss_g_sum / count).item()
    avg_loss_d = (loss_d_sum / count).item()
    
    epoch_time = time.time() - epoch_start
    
    # Logging en proceso principal
    if rank == 0:
        logging.info(f"⏱️ Época [{epoch+1}/{total_epochs}] completada en {epoch_time:.2f}s")
        logging.info(f"📉 Pérdidas - G: {avg_loss_g:.4f}, D: {avg_loss_d:.4f}")
        
        # Escribir en TensorBoard
        if writer:
            writer.add_scalar("Epoch/Generator_Loss", avg_loss_g, epoch)
            writer.add_scalar("Epoch/Discriminator_Loss", avg_loss_d, epoch)
            writer.add_scalar("Epoch/Learning_Rate_G", model.module.optimizer_G.param_groups[0]['lr'], epoch)
            writer.add_scalar("Epoch/Learning_Rate_D", model.module.optimizer_D.param_groups[0]['lr'], epoch)
            writer.add_scalar("Epoch/Time", epoch_time, epoch)
    
    return {
        'loss_G': avg_loss_g,
        'loss_D': avg_loss_d,
        'time': epoch_time
    }

def setup_ddp(rank, world_size, port):
    """
    Configura el entorno para entrenamiento distribuido.
    
    Parámetros:
        rank: Rango del proceso actual
        world_size: Número total de procesos
        port: Puerto para comunicación
    """
    # Configurar dirección maestra
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Optimizaciones para NCCL (pueden mejorar rendimiento)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Usar algoritmo de árbol para reducción
    
    # Inicializar proceso de grupo
    dist.init_process_group(
        backend="nccl", 
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Configurar dispositivo
    torch.cuda.set_device(rank)
    
    # Optimizaciones para rendimiento
    torch.backends.cudnn.benchmark = True

def cleanup_ddp():
    """Limpia el entorno distribuido."""
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    """
    Función principal para cada trabajador (proceso).
    
    Parámetros:
        rank: Rango del proceso actual
        world_size: Número total de procesos
        args: Argumentos de línea de comandos
    """
    # 1. Configurar entorno distribuido
    setup_ddp(rank, world_size, args.port)
    
    # 2. Configurar logging
    logger = setup_logging(rank)
    
    # 3. Configurar semilla para reproducibilidad
    config.setup_seed(config.RANDOM_SEED + rank)  # Semilla diferente por proceso
    
    # 4. Configurar dispositivo
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        logging.info(f"🚀 Iniciando entrenamiento DDP con {world_size} GPUs")
    
    # 5. Configurar TensorBoard (solo en proceso principal)
    writer = None
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/{config.MODEL_NAME}_ddp_{timestamp}")
    
    # 6. Preparar datasets y dataloaders
    train_transform = get_default_transforms(config.IMG_SIZE)
    val_transform = get_validation_transforms(config.IMG_SIZE)
    
    # Dataset de entrenamiento
    train_dataset = ConfocalDataset(
        lr_dir=config.DATA_PATH,
        transform_lr=train_transform,
        transform_hr=train_transform,
        paired=False,
        part_data=config.PART_DATA
    )
    
    # Sampler distribuido para entrenamiento
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # DataLoader de entrenamiento
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    # Dataset y loader de validación
    val_loader = None
    if os.path.exists(config.VALID_PATH):
        val_dataset = ConfocalDataset(
            lr_dir=config.VALID_PATH,
            transform_lr=val_transform,
            transform_hr=val_transform,
            paired=True
        )
        
        # Sampler distribuido para validación
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        if rank == 0:
            logging.info(f"✅ Dataset de validación cargado: {len(val_dataset)} muestras")
    
    # 7. Crear y envolver modelo con DDP
    model = CycleGANModel(config).to(device)
    model = DDP(
        model, 
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
        broadcast_buffers=False  # Mejorar rendimiento, cuidado si hay BatchNorm
    )
    
    # 8. Cargar checkpoint si existe
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        resume_path = args.resume
        if os.path.exists(resume_path):
            start_epoch, best_loss = load_checkpoint(model, resume_path)
            if rank == 0:
                logging.info(f"📂 Checkpoint cargado: {resume_path}, continuando desde época {start_epoch}")
    
    # 9. Preparar muestras fijas para visualización
    fixed_samples = None
    if rank == 0:
        # Cargar muestras para visualización
        for data in train_loader:
            fixed_samples = {
                'LR': data['LR'][:4].to(device),
                'HR': data['HR'][:4].to(device)
            }
            break
    
    # 10. Entrenamiento principal
    total_epochs = config.EPOCHS
    patience = args.patience
    patience_counter = 0
    
    try:
        for epoch in range(start_epoch, total_epochs):
            # Establecer seed de sampler para asegurar muestras diferentes
            train_sampler.set_epoch(epoch)
            if val_loader and val_sampler:
                val_sampler.set_epoch(epoch)
            
            # Entrenar una época
            metrics = train_epoch(
                epoch, model, train_loader, device, rank, world_size, 
                writer, total_epochs
            )
            
            # Actualizar tasa de aprendizaje
            model.module.update_learning_rate()
            
            # Visualizar resultados (solo en proceso principal)
            if rank == 0 and fixed_samples and ((epoch + 1) % 5 == 0 or epoch == 0):
                visualize_results(model, fixed_samples, writer, epoch, rank)
            
            # Ejecutar validación
            if val_loader and ((epoch + 1) % 5 == 0 or epoch == 0):
                val_metrics = validate_ddp(
                    model, val_loader, device, rank, world_size, 
                    writer, epoch
                )
                if rank == 0:
                    logging.info(f"🔍 Validación: PSNR={val_metrics['psnr']:.4f}, SSIM={val_metrics['ssim']:.4f}")
            
            # Verificar si es mejor modelo
            is_best = metrics['loss_G'] < best_loss
            if is_best:
                best_loss = metrics['loss_G']
                patience_counter = 0
                if rank == 0:
                    save_checkpoint(model, epoch, config, best=True)
            else:
                patience_counter += 1
            
            # Guardar checkpoint periódico
            if rank == 0 and (epoch + 1) % config.SAVE_FREQ == 0:
                save_checkpoint(model, epoch, config)
            
            # Early stopping
            if patience > 0 and patience_counter >= patience:
                if rank == 0:
                    logging.info(f"🛑 Early stopping tras {patience} épocas sin mejora")
                break
            
            # Sincronizar procesos al final de cada época
            dist.barrier()
    
    except Exception as e:
        logging.error(f"❌ Error durante entrenamiento: {str(e)}")
        raise e
    
    finally:
        # Limpiar recursos
        if writer:
            writer.close()
        cleanup_ddp()
        
        if rank == 0:
            logging.info("✅ Entrenamiento DDP completado")

# En train_ddp.py
def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrenamiento distribuido de CycleGAN")
    parser.add_argument('--resume', type=str, default='', help='Ruta al checkpoint para continuar')
    parser.add_argument('--patience', type=int, default=20, help='Paciencia para early stopping')
    parser.add_argument('--port', type=int, default=29500, help='Puerto para comunicación distribuida')
    args = parser.parse_args()
    
    # Con torchrun, estas variables ya están establecidas en el entorno
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Iniciar el worker directamente con el rango proporcionado por torchrun
    main_worker(rank, world_size, args)

if __name__ == "__main__":
    main()