"""
train.py

Script principal para entrenar el modelo CycleGAN con SwinIR y ResNet para imágenes confocales.
Este script carga el dataset no pareado, inicializa el modelo, ejecuta el ciclo de entrenamiento,
registra la evolución de las pérdidas usando TensorBoard y guarda los mejores modelos según la pérdida del generador.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import ConfocalDataset, get_default_transforms
import config
from models.cyclegan_model import CycleGANModel

def save_checkpoint(model, epoch, config, best=False):
    """
    Guarda un checkpoint del modelo CycleGAN.
    
    Parámetros:
        model (CycleGANModel): El modelo a guardar.
        epoch (int): La época actual.
        config: Configuración global para obtener la ruta de checkpoint.
        best (bool): Si True, guarda el checkpoint como el mejor modelo.
    """
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)
    
    if best:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_best.pth")
    else:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f"{config.MODEL_NAME}_epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'netG_state_dict': model.netG.state_dict(),
        'netF_state_dict': model.netF.state_dict(),
        'netD_HR_state_dict': model.netD_HR.state_dict(),
        'netD_LR_state_dict': model.netD_LR.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint guardado en: {checkpoint_path}")

def train():
    # Configuración inicial y semilla
    config.setup_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"Entrenamiento en dispositivo: {device}")
    
    # Inicializar TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir="runs/" + config.MODEL_NAME)
    
    # Crear transformaciones para el dataset
    transform = get_default_transforms(config.IMG_SIZE)
    
        # La inicialización correcta del dataset
    dataset = ConfocalDataset(
        lr_dir=config.DATA_PATH,      # No usa npz_file sino lr_dir según la definición
        transform_lr=transform,          # Transformaciones para LR
        transform_hr=transform,          # Transformaciones para HR
        paired=False,                     # Dataset no pareado para entrenamiento
        part_data=config.PART_DATA      # Porcentaje de datos a usar
    )

    # El dataloader está bien definido
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS, 
        drop_last=True
    )
        
    # Inicializar el modelo CycleGAN
    model = CycleGANModel(config)
    model.to(device)
    
    total_epochs = config.EPOCHS
    print(f"Total de épocas: {total_epochs}")
    
    best_loss_G = float('inf')
    
    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        running_loss_G = 0.0
        running_loss_D = 0.0
        count = 0
        
        for i, data in enumerate(dataloader):
            count += 1
            # Configurar las imágenes LR y HR en el modelo
            model.set_input(data)
            
            # Optimización: actualizar generadores y discriminadores
            model.optimize_parameters()
            
            # Obtener las pérdidas actuales para monitoreo
            losses = model.get_current_losses()
            running_loss_G += losses['loss_G']
            running_loss_D += losses['loss_D']
            
            if (i + 1) % config.PRINT_FREQ == 0:
                print(f"Época [{epoch}/{total_epochs}], Iteración [{i+1}/{len(dataloader)}]: "
                      f"Loss_G: {losses['loss_G']:.4f}, Loss_D: {losses['loss_D']:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss_G = running_loss_G / count
        avg_loss_D = running_loss_D / count
        print(f"Época [{epoch}/{total_epochs}] completada en {epoch_time:.2f} segundos. "
              f"Pérdidas promedio - Generadores: {avg_loss_G:.4f}, Discriminadores: {avg_loss_D:.4f}")
        
        # Registrar en TensorBoard
        writer.add_scalar("Loss/Generator", avg_loss_G, epoch)
        writer.add_scalar("Loss/Discriminator", avg_loss_D, epoch)
        writer.add_scalar("Time/Epoch", epoch_time, epoch)
        
        # Guardar checkpoint periódico
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint(model, epoch, config, best=False)
            
        # Guardar el mejor modelo según la pérdida del generador
        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            save_checkpoint(model, epoch, config, best=True)
            print(f"Nuevo mejor modelo guardado en la época {epoch} con Loss_G: {avg_loss_G:.4f}")
            
        # (Opcional) Actualizar learning rate usando schedulers si se han definido
        model.update_learning_rate()
    
    writer.close()

if __name__ == "__main__":
    train()
