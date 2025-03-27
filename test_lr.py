import numpy as np

# Ruta al archivo NPZ
npz_path = 'data/train.npz'

# Cargar los datos con mmap_mode para evitar saturar la memoria
npz_data = np.load(npz_path, mmap_mode='r')
lr_images = npz_data['arr_0']
hr_images = npz_data['arr_1']

# Elegir aleatoriamente 10 índices sin cargar todo a memoria
num_samples = 10

lr_indices = np.random.choice(len(lr_images), num_samples, replace=False)
hr_indices = np.random.choice(len(hr_images), num_samples, replace=False)

# Calcular min/max sin cargar todo en memoria
lr_min, lr_max = float('inf'), float('-inf')
for idx in lr_indices:
    img = lr_images[idx]
    lr_min = min(lr_min, img.min())
    lr_max = max(lr_max, img.max())

hr_min, hr_max = float('inf'), float('-inf')
for idx in hr_indices:
    img = hr_images[idx]
    hr_min = min(hr_min, img.min())
    hr_max = max(hr_max, img.max())

print("Rangos para imágenes LR (baja resolución):")
print(f"  Mínimo: {lr_min}")
print(f"  Máximo: {lr_max}")

print("\nRangos para imágenes HR (alta resolución):")
print(f"  Mínimo: {hr_min}")
print(f"  Máximo: {hr_max}")

# Sugerencia de normalización
print("\nSugerencia de normalización para LR:")
print(f"  Normalize(mean={[lr_min]}, std={[lr_max - lr_min]})")

print("\nSugerencia de normalización para HR:")
print(f"  Normalize(mean={[hr_min]}, std={[hr_max - hr_min]})")
