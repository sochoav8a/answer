import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def visualize_single_image(dataset_path, index=0):
    """
    Carga y visualiza una sola imagen del archivo NPZ para minimizar el uso de memoria.
    
    Parámetros:
    -----------
    dataset_path : str
        Ruta al archivo NPZ que contiene las imágenes.
    index : int
        Índice de la imagen a visualizar.
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(dataset_path):
            print(f"Error: El archivo {dataset_path} no existe.")
            return
        
        print(f"\nCargando archivo NPZ: {dataset_path}")
        print(f"Visualizando sólo la imagen en el índice {index}")
        
        # Abrir el archivo pero sin cargar los datos completos
        with np.load(dataset_path, mmap_mode='r') as data:
            # Solo verificar las formas sin cargar los datos
            print(f"Claves disponibles: {list(data.keys())}")
            print(f"Forma de arr_0 (LR): {data['arr_0'].shape}")
            print(f"Forma de arr_1 (HR): {data['arr_1'].shape}")
            
            # Cargar una sola imagen de cada array
            try:
                # Usamos .copy() para evitar problemas de memoria compartida
                lr_image = data['arr_0'][index].copy()
                print(f"Imagen LR cargada correctamente con forma: {lr_image.shape}")
                
                # Normalizar para visualización si es necesario
                if lr_image.max() > 1.0:
                    lr_image = lr_image / 255.0
                
                # Visualizar solo la imagen LR primero
                plt.figure(figsize=(6, 6))
                plt.imshow(lr_image)
                plt.title(f"LR (arr_0) - Índice: {index}")
                plt.axis('off')
                plt.tight_layout()
                
                output_filename = f"imagen_lr_{os.path.basename(dataset_path)}_{index}.png"
                plt.savefig(output_filename)
                print(f"Imagen LR guardada como: {output_filename}")
                plt.close()  # Cerrar para liberar memoria
                
                # Ahora intentar cargar la imagen HR
                hr_image = data['arr_1'][index].copy()
                print(f"Imagen HR cargada correctamente con forma: {hr_image.shape}")
                
                if hr_image.max() > 1.0:
                    hr_image = hr_image / 255.0
                
                plt.figure(figsize=(6, 6))
                plt.imshow(hr_image)
                plt.title(f"HR (arr_1) - Índice: {index}")
                plt.axis('off')
                plt.tight_layout()
                
                output_filename = f"imagen_hr_{os.path.basename(dataset_path)}_{index}.png"
                plt.savefig(output_filename)
                print(f"Imagen HR guardada como: {output_filename}")
                plt.close()  # Cerrar para liberar memoria
                
            except Exception as e:
                print(f"Error al cargar la imagen individual: {e}")
    
    except Exception as e:
        print(f"Error general: {e}")

def verify_dataset_structure(dataset_path):
    """
    Verifica sólo la estructura del archivo NPZ sin cargar los datos.
    
    Parámetros:
    -----------
    dataset_path : str
        Ruta al archivo NPZ.
    """
    try:
        # Verificar que el archivo existe
        if not os.path.exists(dataset_path):
            print(f"Error: El archivo {dataset_path} no existe.")
            return False
            
        # Intentar abrir el archivo para verificar su estructura
        with np.load(dataset_path, mmap_mode='r') as data:
            # Verificar que contiene las claves necesarias
            if 'arr_0' not in data or 'arr_1' not in data:
                print(f"Error: El archivo no contiene las claves 'arr_0' y 'arr_1'")
                return False
                
            # Imprimir información básica
            print(f"\nVerificación de estructura: {dataset_path}")
            print(f"Claves disponibles: {list(data.keys())}")
            print(f"Forma de arr_0 (LR): {data['arr_0'].shape}")
            print(f"Forma de arr_1 (HR): {data['arr_1'].shape}")
            
            # Verificar las dimensiones
            if len(data['arr_0'].shape) != 4 or len(data['arr_1'].shape) != 4:
                print("Advertencia: Las matrices no tienen la forma esperada (N, H, W, C)")
            
            return True
            
    except Exception as e:
        print(f"Error al verificar la estructura del archivo: {e}")
        return False

if __name__ == "__main__":
    # Rutas a los archivos NPZ
    train_path = 'data/train.npz'
    val_path = 'data/val.npz'
    
    # Verificar sólo la estructura de los archivos
    print("=== Verificando estructura de los archivos NPZ ===")
    verify_dataset_structure(train_path)
    verify_dataset_structure(val_path)
    
    # Visualizar una sola imagen de cada archivo
    print("\n=== Visualizando imágenes individuales ===")
    visualize_single_image(train_path, index=0)
    visualize_single_image(val_path, index=0)
    
    print("\nNota: Si necesitas visualizar más imágenes, ejecuta el script")
    print("nuevamente con diferentes valores de índice.")