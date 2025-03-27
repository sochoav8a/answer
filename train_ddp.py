# train_ddp.py
"""
Este script entrena el modelo CycleGAN usando DistributedDataParallel (DDP) con mejoras de rendimiento y estabilidad.
"""
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from data.dataset import ConfocalDataset, get_default_transforms
from models.cyclegan_model import CycleGANModel

def save_checkpoint(model, epoch, config, best=False):
    """Guarda checkpoints optimizados para DDP con verificación de rank 0."""
    if dist.get_rank() != 0:
        return
    
    if not os.path.exists(config.CHECKPOINTS_DIR):
        os.makedirs(config.CHECKPOINTS_DIR)
    
    checkpoint_name = f"{config.MODEL_NAME}_best_{epoch}.pth" if best else f"{config.MODEL_NAME}_epoch_{epoch}.pth"
    checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, checkpoint_name)
    
    # Guardado optimizado con estado consolidado
    state = {
        'epoch': epoch,
        'netG': model.module.netG.state_dict(),
        'netF': model.module.netF.state_dict(),
        'netD_HR': model.module.netD_HR.state_dict(),
        'netD_LR': model.module.netD_LR.state_dict(),
        'optimizer_G': model.module.optimizer_G.state_dict(),
        'optimizer_D': model.module.optimizer_D.state_dict(),
        'scaler_G': model.module.scaler_G.state_dict(),
        'scaler_D': model.module.scaler_D.state_dict(),
    }
    
    torch.save(state, checkpoint_path)
    print(f"Checkpoint guardado en: {checkpoint_path}")

def train_one_epoch(model, dataloader, epoch, total_epochs, writer, local_rank, world_size):
    """Época de entrenamiento con sincronización cross-GPU y manejo mejorado de métricas."""
    model.train()
    start_time = time.time()
    total_loss_G = torch.tensor(0.0, device=local_rank)
    total_loss_D = torch.tensor(0.0, device=local_rank)
    count = torch.tensor(0, device=local_rank)
    
    # Configurar barra de progreso solo en rank 0
    progress_bar = dataloader
    if local_rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}]", 
                          leave=False, dynamic_ncols=True)

    for data in progress_bar:
        count += 1
        
        # Forward y optimización
        model.module.set_input(data)
        model.module.optimize_parameters()

        # Acumular pérdidas sincronizadas
        losses = model.module.get_current_losses()
        total_loss_G += torch.tensor(losses['loss_G'], device=local_rank)
        total_loss_D += torch.tensor(losses['loss_D'], device=local_rank)

        # Actualizar barra de progreso
        if local_rank == 0:
            progress_bar.set_postfix({
                'loss_G': f"{losses['loss_G']:.4f}",
                'loss_D': f"{losses['loss_D']:.4f}",
                'lr': f"{model.module.optimizer_G.param_groups[0]['lr']:.2e}"
            })

    # Sincronizar métricas entre GPUs
    dist.all_reduce(total_loss_G, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_loss_D, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    avg_loss_G = (total_loss_G / count).item()
    avg_loss_D = (total_loss_D / count).item()
    epoch_time = time.time() - start_time

    # Logging en rank 0
    if local_rank == 0:
        print(f"\nÉpoca [{epoch}/{total_epochs}] Tiempo: {epoch_time:.2f}s")
        print(f"Pérdidas - G: {avg_loss_G:.4f}, D: {avg_loss_D:.4f}")
        
        if writer:
            writer.add_scalar("Loss/Generator", avg_loss_G, epoch)
            writer.add_scalar("Loss/Discriminator", avg_loss_D, epoch)
            writer.add_scalar("LearningRate/Generator", model.module.optimizer_G.param_groups[0]['lr'], epoch)
            writer.add_scalar("LearningRate/Discriminator", model.module.optimizer_D.param_groups[0]['lr'], epoch)
            writer.add_scalar("Time/Epoch", epoch_time, epoch)

    return avg_loss_G, avg_loss_D

def main_worker(local_rank, world_size):
    """Worker principal con inicialización mejorada y manejo de recursos."""
    # 1. Inicialización DDP
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank
    )
    
    # 2. Configuración determinística
    config.setup_seed(config.RANDOM_SEED + local_rank)
    torch.backends.cudnn.benchmark = True  # Solo si los tamaños de entrada son fijos

    # 3. Preparación de datos
    transform = get_default_transforms(config.IMG_SIZE)
    dataset = ConfocalDataset(
        lr_dir=config.DATA_PATH,
        transform_lr=transform,
        transform_hr=transform,
        paired=False,
        part_data=config.PART_DATA
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.NUM_WORKERS > 0
    )

    # 4. Inicialización del modelo
    model = CycleGANModel(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 5. Entrenamiento
    best_loss = float('inf')
    writer = SummaryWriter(log_dir="runs/" + config.MODEL_NAME + "_ddp") if local_rank == 0 else None
    
    try:
        for epoch in range(1, config.EPOCHS + 1):
            sampler.set_epoch(epoch)
            
            avg_loss_G, avg_loss_D = train_one_epoch(
                model, dataloader, epoch, config.EPOCHS, writer, local_rank, world_size
            )
            
            model.module.update_learning_rate()

            # Guardar checkpoints
            if local_rank == 0:
                if avg_loss_G < best_loss:
                    best_loss = avg_loss_G
                    save_checkpoint(model, epoch, config, best=True)
                
                if epoch % config.SAVE_FREQ == 0:
                    save_checkpoint(model, epoch, config)

    finally:
        # Limpieza final
        if writer:
            writer.close()
        dist.destroy_process_group()

def train_ddp():
    """Punto de entrada principal para el entrenamiento DDP."""
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Optimización de comunicación NCCL
    os.environ["NCCL_ALGO"] = "Tree"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_BUFFSIZE"] = "2097152"
    
    main_worker(local_rank, world_size)

if __name__ == "__main__":
    train_ddp()