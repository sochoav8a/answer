import numpy as np
import matplotlib.pyplot as plt

def analyze_and_fix_npz(npz_path):
    # Cargar datos
    data = np.load(npz_path)
    lr = data['arr_0']
    hr = data['arr_1']
    
    print("=== Análisis Inicial ===")
    print(f"LR - dtype: {lr.dtype} | Forma: {lr.shape} | Rango: [{lr.min()}, {lr.max()}]")
    print(f"HR - dtype: {hr.dtype} | Forma: {hr.shape} | Rango: [{hr.min()}, {hr.max()}]")
    
    # Detectar problema común
    if lr.dtype == 'float32' and lr.max() > 1.0:
        print("\n¡Problema detectado!: Imágenes float32 en rango 0-255 en lugar de 0-1")
        
        # Conversión a uint8
        lr_uint8 = lr.astype(np.uint8)
        hr_uint8 = hr.astype(np.uint8)
        
        # Guardar versión corregida
        new_path = npz_path.replace('.npz', '_fixed.npz')
        np.savez(new_path, arr_0=lr_uint8, arr_1=hr_uint8)
        
        print(f"\nArchivo corregido guardado en: {new_path}")
        print("=== Nuevos valores ===")
        print(f"LR - dtype: {lr_uint8.dtype} | Rango: [{lr_uint8.min()}, {lr_uint8.max()}]")
        print(f"HR - dtype: {hr_uint8.dtype} | Rango: [{hr_uint8.min()}, {hr_uint8.max()}]")
        
        return new_path
    else:
        print("\nEl archivo NPZ ya tiene formato correcto")
        return npz_path

def visualize_samples(npz_path, num_samples=3):
    data = np.load(npz_path)
    lr = data['arr_0']
    hr = data['arr_1']
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = np.random.randint(len(lr))
        
        # LR
        plt.subplot(2, num_samples, i+1)
        if lr.dtype == 'uint8':
            plt.imshow(lr[idx])
        else:
            plt.imshow(lr[idx].astype(int))  # Forzar a valores enteros
        plt.title(f"LR (dtype: {lr.dtype})\nMin: {lr[idx].min()} Max: {lr[idx].max()}")
        plt.axis('off')
        
        # HR
        plt.subplot(2, num_samples, i+1+num_samples)
        if hr.dtype == 'uint8':
            plt.imshow(hr[idx])
        else:
            plt.imshow(hr[idx].astype(int))
        plt.title(f"HR (dtype: {hr.dtype})\nMin: {hr[idx].min()} Max: {hr[idx].max()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Uso
original_npz = 'val.npz'
fixed_npz = analyze_and_fix_npz(original_npz)
visualize_samples(fixed_npz)