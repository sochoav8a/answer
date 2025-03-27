"""
Script de prueba para validar la carga correcta de imágenes desde archivos NPZ
Muestra una imagen LR y una HR de cada conjunto (train y val)
"""

import torch
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Asegúrate de que el módulo esté en el path
sys.path.append('.')

# Importamos nuestro dataset
from data.dataset import ConfocalDataset, get_default_transforms, get_validation_transforms

def denormalize(tensor):
    """
    Convierte un tensor normalizado en [-1,1] a [0,1] para visualización
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def visualize_image_tensor(tensor, title):
    """
    Visualiza un tensor de imagen
    """
    # Convertimos el tensor a numpy para matplotlib
    if tensor.dim() == 4:  # Si es un batch, tomamos la primera imagen
        tensor = tensor[0]
    
    # Convertimos de (C,H,W) a (H,W,C)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Aseguramos que esté en rango [0,1]
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_train_dataset():
    """
    Prueba el dataset de entrenamiento (no pareado)
    """
    print("\n--- PROBANDO DATASET DE ENTRENAMIENTO (NO PAREADO) ---")
    
    # Cargamos el dataset de entrenamiento
    train_transforms = get_default_transforms(img_size=256)
    train_dataset = ConfocalDataset(
        lr_dir='data/train.npz',  # Contiene arr_0 (LR) y arr_1 (HR)
        transform_lr=train_transforms,
        transform_hr=train_transforms,
        paired=False  # No pareado para entrenamiento
    )
    
    # Obtenemos un ejemplo
    sample = train_dataset[0]
    
    print(f"Forma del tensor LR: {sample['LR'].shape}")
    print(f"Forma del tensor HR: {sample['HR'].shape}")
    print(f"Índice de la imagen LR: {sample['LR_idx']}")
    print(f"Índice de la imagen HR: {sample['HR_idx']}")
    
    # Visualizamos
    lr_img = denormalize(sample['LR'])
    hr_img = denormalize(sample['HR'])
    
    print(f"Rango de valores LR: [{lr_img.min():.4f}, {lr_img.max():.4f}]")
    print(f"Rango de valores HR: [{hr_img.min():.4f}, {hr_img.max():.4f}]")
    
    print("\nEn un dataset NO PAREADO, el índice de HR suele ser aleatorio")
    
    return lr_img, hr_img, sample['LR_idx'], sample['HR_idx']

def test_val_dataset():
    """
    Prueba el dataset de validación (pareado)
    """
    print("\n--- PROBANDO DATASET DE VALIDACIÓN (PAREADO) ---")
    
    # Cargamos el dataset de validación
    val_transforms = get_validation_transforms(img_size=256)
    val_dataset = ConfocalDataset(
        lr_dir='data/val.npz',  # Contiene arr_0 (LR) y arr_1 (HR)
        transform_lr=val_transforms,
        transform_hr=val_transforms,
        paired=True  # Pareado para validación
    )
    
    # Obtenemos un ejemplo
    sample = val_dataset[0]
    
    print(f"Forma del tensor LR: {sample['LR'].shape}")
    print(f"Forma del tensor HR: {sample['HR'].shape}")
    print(f"Índice de la imagen LR: {sample['LR_idx']}")
    print(f"Índice de la imagen HR: {sample['HR_idx']}")
    
    # Visualizamos
    lr_img = denormalize(sample['LR'])
    hr_img = denormalize(sample['HR'])
    
    print(f"Rango de valores LR: [{lr_img.min():.4f}, {lr_img.max():.4f}]")
    print(f"Rango de valores HR: [{hr_img.min():.4f}, {hr_img.max():.4f}]")
    
    print("\nEn un dataset PAREADO, el índice de LR y HR debe ser el mismo")
    assert sample['LR_idx'] == sample['HR_idx'], "¡Error! Los índices deberían ser iguales en modo pareado"
    
    return lr_img, hr_img, sample['LR_idx'], sample['HR_idx']

def inspect_npz_file(file_path):
    """
    Inspecciona el contenido de un archivo NPZ
    """
    print(f"\n--- INSPECCIONANDO ARCHIVO NPZ: {file_path} ---")
    try:
        data = np.load(file_path)
        print(f"Claves en el archivo: {list(data.keys())}")
        
        if 'arr_0' in data:
            arr0 = data['arr_0']
            print(f"arr_0 (LR) - Forma: {arr0.shape}, Tipo: {arr0.dtype}")
            print(f"arr_0 (LR) - Rango de valores: [{arr0.min()}, {arr0.max()}]")
        
        if 'arr_1' in data:
            arr1 = data['arr_1']
            print(f"arr_1 (HR) - Forma: {arr1.shape}, Tipo: {arr1.dtype}")
            print(f"arr_1 (HR) - Rango de valores: [{arr1.min()}, {arr1.max()}]")
    
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")

if __name__ == "__main__":
    # Primero inspeccionamos los archivos NPZ
    inspect_npz_file('data/train.npz')
    inspect_npz_file('data/val.npz')
    
    # Probamos el dataset de entrenamiento
    lr_train, hr_train, lr_idx_train, hr_idx_train = test_train_dataset()
    
    # Probamos el dataset de validación
    lr_val, hr_val, lr_idx_val, hr_idx_val = test_val_dataset()
    
    # Mensaje de éxito
    print("\n¡Prueba completada con éxito!")
    print("El dataset está cargando correctamente las imágenes desde los archivos NPZ.")