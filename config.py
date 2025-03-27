"""
config.py

Configuración completa y centralizada para el proyecto CycleGAN+SwinIR:
- Rutas y parámetros para datos
- Configuración de entrenamiento
- Hiperparámetros del modelo
- Opciones para single-GPU y multi-GPU (DDP)
- Configuración de optimizadores y pérdidas
- Opciones de visualización
"""

import os
import math
import torch
import random
import numpy as np
from datetime import datetime

#==============================================================================
# RUTAS Y CONFIGURACIÓN DE DATOS
#==============================================================================

# Rutas de datos
DATA_PATH = 'data/train_fixed.npz'   # Archivo NPZ con imágenes LR (arr_0) y HR (arr_1)
VALID_PATH = 'data/val_fixed.npz'    # Archivo NPZ de validación con LR (arr_0) y HR (arr_1)
PART_DATA = 0.001                   # Porcentaje de datos a usar para entrenamiento (1.0 = 100%)

# Parámetros del dataset
IMG_SIZE = 256                       # Resolución de las imágenes (en píxeles) - debe ser múltiplo de WINDOW_SIZE
IMG_CHANNELS = 3                     # Número de canales de las imágenes (3 para RGB, 1 para grayscale)
NORMALIZE_RANGE = (-1, 1)            # Rango de normalización para imágenes - recomendado (-1, 1) para GANs

# Parámetros de procesamiento
BATCH_SIZE = 1                       # Tamaño de lote - ajustar según memoria GPU disponible
NUM_WORKERS = 1                   # Número de subprocesos para carga de datos (recomendado: 4 × núcleos)
PIN_MEMORY = True                    # Acelera transferencia CPU→GPU (mantener True si se usa GPU)
PREFETCH_FACTOR = 2                  # Factor de precarga para dataloader (PyTorch >= 1.7.0)

#==============================================================================
# PARÁMETROS DE ENTRENAMIENTO
#==============================================================================

# General
EPOCHS = 100                         # Número total de épocas
SAVE_FREQ = 5                        # Guardar checkpoints cada N épocas
PRINT_FREQ = 100                     # Frecuencia para imprimir pérdidas (iteraciones)
VALIDATE_FREQ = 5                    # Ejecutar validación cada N épocas
VISUALIZE_FREQ = 5                   # Generar visualizaciones cada N épocas

# Learning rate y schedulers
LR_GENERATOR = 2e-4                  # Tasa de aprendizaje para generadores
LR_DISCRIMINATOR = 1e-4              # Tasa de aprendizaje para discriminadores
BETA1 = 0.5                          # Parámetro beta1 para Adam (momentum) - 0.5 bueno para GANs
BETA2 = 0.999                        # Parámetro beta2 para Adam (RMSprop)
WEIGHT_DECAY = 1e-5                  # Regularización L2 para prevenir sobreajuste
LR_POLICY = 'linear'                 # Política de decaimiento: 'linear', 'cosine', 'step', 'plateau'
WARMUP_EPOCHS = 5                    # Épocas de calentamiento con aumento gradual de LR
LR_DECAY_START = 50                  # Época en la que comienza el decaimiento (para linear)
LR_DECAY_FACTOR = 0.1                # Factor para decaimiento step (para step y plateau)
LR_DECAY_STEPS = 20                  # Intervalo de decaimiento para step policy

# Optimización y estabilidad
USE_AMP = True                       # Usar precisión mixta automática para ahorrar memoria y acelerar
CLIP_GRADIENT_NORM = 1.0             # Valor máximo para recorte de gradientes (0 para desactivar)
R1_GAMMA = 10.0                      # Factor para regularización R1 en discriminadores (0 para desactivar)
POOL_SIZE = 50                       # Tamaño del buffer para imágenes históricas (estabiliza discriminador)
EMA_DECAY = 0.999                    # Factor de decaimiento para Exponential Moving Average (0 para desactivar)

# Pesos de las pérdidas
LAMBDA_CYCLE = 10.0                  # Peso para pérdida de ciclo (reconstrucción)
LAMBDA_IDENTITY = 5.0                # Peso para pérdida de identidad (preservación de contenido)
LAMBDA_PERCEPTUAL = 1.0              # Peso para pérdida perceptual (características VGG)
LAMBDA_GRADIENT = 0.0                # Peso para pérdida de gradiente de imagen (preservar bordes)

# Parámetros avanzados
CONSISTENCY_TRAINING = False         # Usar entrenamiento con consistencia (regularización adicional)
LABEL_SMOOTHING = 0.1                # Suavizado de etiquetas para discriminador (estabilidad)
DIFF_AUGMENT = False                 # Usar aumentación diferenciable (para datasets pequeños)
USE_SPECTRAL_NORM = True             # Usar normalización espectral en discriminadores

#==============================================================================
# PARÁMETROS DEL MODELO
#==============================================================================

# Parámetros optimizados para SwinIR
SWINIR_WINDOW_SIZE = 8               # Tamaño de ventana para atención - mayor para más contexto
SWINIR_DEPTHS = [2]            # Profundidad de cada nivel - más niveles/bloques = más capacidad
SWINIR_CHANNELS = 30       # Canales base - más canales = más capacidad pero más memoria
SWINIR_NUM_HEADS = 8                 # Número de cabezas de atención - recomendado divisor de CHANNELS
SWINIR_MLP_RATIO = 4.0               # Ratio de expansión para MLP - mayor valor = más capacidad
SWINIR_DROPOUT = 0.1                 # Tasa de dropout para regularización

# Parámetros para generador ResNet
RESNET_BLOCKS = 9                    # Número de bloques residuales para el generador ResNet
RESNET_NGF = 64                      # Número de filtros base para generador ResNet

# Parámetros para discriminador
DISC_NDF = 64                        # Número de filtros base para discriminador
DISC_N_LAYERS = 3                    # Número de capas en discriminador PatchGAN

#==============================================================================
# ENTRENAMIENTO DISTRIBUIDO (DDP)
#==============================================================================

# Parámetros DDP
DDP_BACKEND = 'nccl'                 # Backend para comunicación inter-GPU ('nccl' para NVIDIA, 'gloo' para CPU)
DDP_PORT = 29500                     # Puerto para comunicación
DDP_FIND_UNUSED_PARAMS = True        # Buscar parámetros no utilizados en forward (para modelos complejos)
DDP_BROADCAST_BUFFERS = False        # Sincronizar buffers entre procesos (False mejora rendimiento)
DDP_OPTIMIZE_COMM = True             # Optimizaciones para comunicación NCCL

#==============================================================================
# PATHS Y DIRECTORIOS
#==============================================================================

# Directorio raíz para organizar resultados
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"run_{TIMESTAMP}"

# Checkpoints, resultados y logs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', RUN_NAME)
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_NAME = 'cycle_swinir'

# Crear directorios si no existen
for dir_path in [CHECKPOINTS_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

#==============================================================================
# DEVICE CONFIGURATION
#==============================================================================

# Detección automática de dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', None)

# Configuraciones de precision
USE_FP16 = True if USE_AMP else False    # Usar FP16 (con AMP)
CUDNN_BENCHMARK = True                   # Optimizar para tamaños fijos de entrada
CUDNN_DETERMINISTIC = False              # False para mejor rendimiento, True para reproducibilidad

#==============================================================================
# PARÁMETROS DE EVALUACIÓN
#==============================================================================

# Métricas y visualización
CALC_METRICS = True                  # Calcular PSNR/SSIM/FID si hay pares disponibles
SAVE_VISUALS = True                  # Guardar imágenes de resultados durante validación
MAX_VISUALS = 16                     # Número máximo de muestras a visualizar
METRICS_WINDOW = 11                  # Tamaño de ventana para métricas como SSIM

#==============================================================================
# CONFIGURACIÓN DE REPRODUCIBILIDAD
#==============================================================================

# Semilla aleatoria
RANDOM_SEED = 42

def setup_seed(seed):
    """
    Configura semillas aleatorias para reproducibilidad.
    
    Args:
        seed (int): Valor de semilla a utilizar
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    if CUDNN_DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK

def get_lr_scheduler(optimizer, policy=LR_POLICY):
    """
    Crea un scheduler de tasa de aprendizaje según la política especificada.
    
    Args:
        optimizer: Optimizador de PyTorch
        policy (str): Tipo de política de LR
        
    Returns:
        scheduler: Objeto scheduler de PyTorch
    """
    if policy == 'linear':
        # Decaimiento lineal desde la época LR_DECAY_START
        def lambda_rule(epoch):
            if epoch < WARMUP_EPOCHS:
                return epoch / WARMUP_EPOCHS
            elif epoch < LR_DECAY_START:
                return 1.0
            else:
                return 1.0 - min(1.0, (epoch - LR_DECAY_START) / (EPOCHS - LR_DECAY_START))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        
    elif policy == 'cosine':
        # Decaimiento coseno con calentamiento
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=LR_DECAY_STEPS, T_mult=2, eta_min=1e-6
        )
        
    elif policy == 'step':
        # Decaimiento escalonado
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_DECAY_STEPS, gamma=LR_DECAY_FACTOR
        )
        
    elif policy == 'plateau':
        # Reducción en plateau (útil con validación)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_DECAY_FACTOR, patience=5,
            threshold=0.01, min_lr=1e-6
        )
        
    else:
        raise ValueError(f"Política de LR no soportada: {policy}")
        
    return scheduler

def get_optimizer(parameters, lr, optimizer_type='adam'):
    """
    Crea un optimizador según el tipo especificado.
    
    Args:
        parameters: Parámetros a optimizar
        lr (float): Tasa de aprendizaje
        optimizer_type (str): Tipo de optimizador
        
    Returns:
        optimizer: Objeto optimizador de PyTorch
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            parameters, lr=lr, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            parameters, lr=lr, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_type}")

# Configurar semilla global al importar el módulo
setup_seed(RANDOM_SEED)

# Mostrar configuración de dispositivo
print(f"Configuración PyTorch: CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}" 
      if torch.cuda.is_available() else "Configuración PyTorch: CPU")
print(f"Dispositivo seleccionado: {DEVICE} | GPUs disponibles: {NUM_GPUS}")