import numpy as np
import matplotlib.pyplot as plt

def analyze_image_ranges(npz_path, num_samples=5):
    # Cargar el archivo NPZ
    data = np.load(npz_path)
    
    # Verificar que existan las claves
    if 'arr_0' not in data or 'arr_1' not in data:
        raise ValueError("El archivo NPZ debe contener 'arr_0' (LR) y 'arr_1' (HR)")
    
    lr_images = data['arr_0']
    hr_images = data['arr_1']
    
    print(f"Total imágenes LR: {lr_images.shape[0]}")
    print(f"Total imágenes HR: {hr_images.shape[0]}")
    print(f"\nFormato LR: {lr_images.shape}, dtype: {lr_images.dtype}")
    print(f"Formato HR: {hr_images.shape}, dtype: {hr_images.dtype}")
    
    def get_stats(images, name):
        # Seleccionar muestras aleatorias
        sample_indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        samples = images[sample_indices]
        
        print(f"\n--- Estadísticas {name} ---")
        for i, img in enumerate(samples):
            print(f"Imagen {i+1}:")
            print(f"  Formato: {img.shape}")
            print(f"  Mín: {np.min(img):.2f}, Máx: {np.max(img):.2f}")
            print(f"  Media: {np.mean(img):.2f}, Desv: {np.std(img):.2f}")
            
            # Mostrar imagen
            plt.figure()
            if len(img.shape) == 2:  # Grayscale
                plt.imshow(img, cmap='gray')
            else:  # Color (asumiendo canales últimos)
                plt.imshow(img)
            plt.title(f"{name} sample {i+1}")
            plt.colorbar()
            plt.show()
    
    # Analizar LR
    get_stats(lr_images, "LR")
    
    # Analizar HR
    get_stats(hr_images, "HR")

if __name__ == "__main__":
    # Ejemplo de uso
    analyze_image_ranges('train_fixed.npz', num_samples=3)