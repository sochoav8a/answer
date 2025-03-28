import numpy as np
import matplotlib.pyplot as plt

def load_and_inspect_npz_files(train_path='train_fixed.npz', val_path='val_fixed.npz'):
    """
    Carga y muestra información detallada de archivos .npz para datasets de super-resolución.
    
    Args:
        train_path: Ruta al archivo de entrenamiento
        val_path: Ruta al archivo de validación
    
    Returns:
        Diccionario con todos los datos cargados
    """
    print("=" * 50)
    print("CARGANDO DATASETS DE SUPER-RESOLUCIÓN")
    print("=" * 50)
    
    # Cargar archivos
    try:
        train_data = np.load(train_path)
        val_data = np.load(val_path)
        
        # Mostrar las claves disponibles
        print(f"\nClaves en {train_path}:", train_data.files)
        print(f"Claves en {val_path}:", val_data.files)
        
        # Información sobre los arrays en train_fixed.npz
        print("\nINFORMACIÓN SOBRE DATASET DE ENTRENAMIENTO:")
        for key in train_data.files:
            arr = train_data[key]
            print(f"  {key}: forma={arr.shape}, tipo={arr.dtype}")
            print(f"       rango=[{arr.min():.4f}, {arr.max():.4f}], media={arr.mean():.4f}")
            
        # Información sobre los arrays en val_fixed.npz
        print("\nINFORMACIÓN SOBRE DATASET DE VALIDACIÓN:")
        for key in val_data.files:
            arr = val_data[key]
            print(f"  {key}: forma={arr.shape}, tipo={arr.dtype}")
            print(f"       rango=[{arr.min():.4f}, {arr.max():.4f}], media={arr.mean():.4f}")
        
        # Contar imágenes en cada conjunto
        num_train_lr = train_data['arr_0'].shape[0]
        num_train_hr = train_data['arr_1'].shape[0]
        num_val_lr = val_data['arr_0'].shape[0]
        num_val_hr = val_data['arr_1'].shape[0]
        
        print("\nRECUENTO DE IMÁGENES:")
        print(f"  Imágenes LR (baja resolución) en entrenamiento: {num_train_lr}")
        print(f"  Imágenes HR (alta resolución) en entrenamiento: {num_train_hr}")
        print(f"  Imágenes LR (baja resolución) en validación: {num_val_lr}")
        print(f"  Imágenes HR (alta resolución) en validación: {num_val_hr}")
        
        # Verificar si las imágenes están pareadas o no
        print("\nVERIFICACIÓN DE PAREADO:")
        print(f"  Dataset de entrenamiento: {'PAREADO' if num_train_lr == num_train_hr else 'NO PAREADO'}")
        print(f"  Dataset de validación: {'PAREADO' if num_val_lr == num_val_hr else 'NO PAREADO'}")
        
        # Mostrar forma de las imágenes
        print("\nFORMA DE LAS IMÁGENES:")
        if len(train_data['arr_0'].shape) >= 3:
            print(f"  LR entrenamiento: {train_data['arr_0'][0].shape}")
        if len(train_data['arr_1'].shape) >= 3:
            print(f"  HR entrenamiento: {train_data['arr_1'][0].shape}")
        if len(val_data['arr_0'].shape) >= 3:
            print(f"  LR validación: {val_data['arr_0'][0].shape}")
        if len(val_data['arr_1'].shape) >= 3:
            print(f"  HR validación: {val_data['arr_1'][0].shape}")
        
        # Preparar el diccionario de datos para devolver
        data_dict = {
            'train_lr': train_data['arr_0'],
            'train_hr': train_data['arr_1'],
            'val_lr': val_data['arr_0'],
            'val_hr': val_data['arr_1'],
        }
        
        print("\nDATOS CARGADOS CORRECTAMENTE")
        print("=" * 50)
        
        return data_dict
        
    except FileNotFoundError as e:
        print(f"ERROR: No se pudo encontrar uno de los archivos. {e}")
    except Exception as e:
        print(f"ERROR al cargar o procesar los archivos: {e}")
        return None

def show_sample_images(data_dict, num_samples=2):
    """
    Muestra algunas imágenes de ejemplo de los datasets.
    
    Args:
        data_dict: Diccionario con los datos cargados
        num_samples: Número de muestras a visualizar
    """
    if data_dict is None:
        print("No hay datos para mostrar.")
        return
        
    plt.figure(figsize=(12, 4 * num_samples))
    
    for i in range(min(num_samples, len(data_dict['train_lr']))):
        # Imagen LR de entrenamiento
        plt.subplot(num_samples, 4, i*4+1)
        if len(data_dict['train_lr'][i].shape) == 3:
            plt.imshow(np.clip(data_dict['train_lr'][i], 0, 1))
        else:
            plt.imshow(np.clip(data_dict['train_lr'][i], 0, 1), cmap='gray')
        plt.title(f"Train LR #{i}")
        
        # Imagen HR de entrenamiento
        plt.subplot(num_samples, 4, i*4+2)
        if len(data_dict['train_hr'][i].shape) == 3:
            plt.imshow(np.clip(data_dict['train_hr'][i], 0, 1))
        else:
            plt.imshow(np.clip(data_dict['train_hr'][i], 0, 1), cmap='gray')
        plt.title(f"Train HR #{i}")
        
        # Imagen LR de validación
        plt.subplot(num_samples, 4, i*4+3)
        if len(data_dict['val_lr'][i].shape) == 3:
            plt.imshow(np.clip(data_dict['val_lr'][i], 0, 1))
        else:
            plt.imshow(np.clip(data_dict['val_lr'][i], 0, 1), cmap='gray')
        plt.title(f"Val LR #{i}")
        
        # Imagen HR de validación
        plt.subplot(num_samples, 4, i*4+4)
        if len(data_dict['val_hr'][i].shape) == 3:
            plt.imshow(np.clip(data_dict['val_hr'][i], 0, 1))
        else:
            plt.imshow(np.clip(data_dict['val_hr'][i], 0, 1), cmap='gray')
        plt.title(f"Val HR #{i}")
    
    plt.tight_layout()
    plt.show()

# Función principal
if __name__ == "__main__":
    # Cargar y examinar los datos
    data = load_and_inspect_npz_files('train_fixed.npz', 'val_fixed.npz')
    
    # Mostrar algunas imágenes de ejemplo (opcional)
    # show_sample_images(data)
    
    print("\nSUGERENCIA PARA USAR ESTOS DATOS:")
    print("""
Para usar estos datos en otro prompt o en un modelo de super-resolución, puedes:

1. Importar este script como módulo:
   from npz_reader import load_and_inspect_npz_files
   
2. Cargar los datos:
   data = load_and_inspect_npz_files('train_fixed.npz', 'val_fixed.npz')
   
3. Acceder a los arrays:
   train_lr = data['train_lr']  # Imágenes de baja resolución para entrenamiento
   train_hr = data['train_hr']  # Imágenes de alta resolución para entrenamiento
   val_lr = data['val_lr']      # Imágenes de baja resolución para validación
   val_hr = data['val_hr']      # Imágenes de alta resolución para validación
""")