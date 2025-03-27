import torch
import random
import numpy as np
# ------------------
# -----------------------------------------------------------
# Archivo: config.py
# Descripción:
# Define parámetros y rutas globales para el entrenamiento y la inferencia.
# Ajusta los valores según tus necesidades.
# -----------------------------------------------------------------------------
# Rutas de datos
DATA_PATH = 'data/train_fixed.npz'  # Archivo NPZ con imágenes LR (arr_0) y HR (arr_1)
VALID_PATH = 'data/val_fixed.npz'    # Archivo NPZ de validación con LR (arr_0) y HR (arr_1)
PART_DATA = 0.01  # % de datos a usar (1.0 = 100%)
# Parámetros de dataset
IMG_SIZE = 256  # Ancho/alto de las imágenes (en píxeles)
BATCH_SIZE = 1   # Tamaño de lote
NUM_WORKERS = 1  # Número de subprocesos para cargar datos

# Parámetros de entrenamiento
EPOCHS = 100  # Número de épocas
SAVE_FREQ = 5  # Guardar pesos cada N épocas
PRINT_FREQ = 100  # Frecuencia para imprimir pérdidas en pantalla (iters)

# Tasas de aprendizaje
LR_GENERATOR = 2e-4  # Learning rate para generadores
LR_DISCRIMINATOR = 1e-4  # Learning rate para discriminadores
BETA1 = 0.5  # Parámetro beta1 para Adam
BETA2 = 0.999  # Parámetro beta2 para Adam

# Pesos de las pérdidas
LAMBDA_CYCLE = 10.0  # Pérdida de ciclo
LAMBDA_IDENTITY = 5.0  # Pérdida de identidad
LAMBDA_PERCEPTUAL = 1.0  # Pérdida perceptual (ajusta este valor según experimentos)


# Parámetros de modelo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parámetros optimizados para SwinIR
SWINIR_WINDOW_SIZE = 8    # Aumentado de 8 a 12 para capturar mayor contexto
SWINIR_DEPTHS = [2]  # Profundidad asimétrica en vez de [2, 2, 2, 2]
SWINIR_CHANNELS = 60       # Aumentado de 60 a 96 para mayor capacidad representativa

# Checkpoints y resultados
CHECKPOINTS_DIR = 'checkpoints/'
RESULTS_DIR = 'results/'
MODEL_NAME = 'cycle_swinir'

# Parámetros de validación/test
CALC_METRICS = True  # Calcular PSNR/SSIM si hay pares disponibles

# Semilla aleatoria (para reproducibilidad)
RANDOM_SEED = 42

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

setup_seed(RANDOM_SEED)