"""
utils/metrics.py
Este módulo define funciones para calcular dos métricas esenciales:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
Estas métricas se utilizarán para evaluar la calidad de las imágenes generadas.
"""
import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calcula el PSNR (Peak Signal-to-Noise Ratio) entre dos imágenes.
    Parámetros:
    img1 (Tensor): Imagen generada, con valores en [0, max_val].
    img2 (Tensor): Imagen de referencia, con valores en [0, max_val].
    max_val (float): Valor máximo que puede tomar un píxel (por defecto 1.0).
    Retorna:
    psnr (float): Valor de PSNR en decibelios.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def gaussian(window_size, sigma):
    """
    Crea un vector gaussiano 1D.
    Parámetros:
    window_size (int): Tamaño de la ventana.
    sigma (float): Desviación estándar.
    Retorna:
    gauss (Tensor): Vector gaussiano normalizado.
    """
    # Crear un tensor de coordenadas
    coords = torch.arange(window_size, dtype=torch.float)
    # Centrar las coordenadas
    coords = coords - window_size//2
    # Calcular la distribución gaussiana de manera vectorizada
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Crea una ventana 2D a partir de un vector gaussiano 1D.
    Parámetros:
    window_size (int): Tamaño de la ventana.
    channel (int): Número de canales de la imagen.
    Retorna:
    window (Tensor): Ventana 2D expandida para convolución, de forma (channel, 1, window_size, window_size).
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim(img1, img2, window_size=11, channel=3, max_val=1.0):
    """
    Calcula el SSIM (Structural Similarity Index Measure) entre dos imágenes.
    Parámetros:
    img1 (Tensor): Imagen generada de forma (B, C, H, W) con valores en [0, max_val].
    img2 (Tensor): Imagen de referencia de forma (B, C, H, W) con valores en [0, max_val].
    window_size (int): Tamaño de la ventana para la convolución (por defecto 11).
    channel (int): Número de canales de la imagen.
    max_val (float): Valor máximo que puede tomar un píxel (por defecto 1.0).
    Retorna:
    ssim (float): SSIM promedio sobre el batch.
    """
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()