"""
models/generator_swinir.py

Implementación mejorada del generador SwinIR para imágenes confocales con:
- Mayor capacidad de representación mediante atención progresiva
- Mecanismos de atención multi-escala
- Conexiones residuales con escalado adaptativo
- Regularización mejorada con stochastic depth avanzado
- Optimización de memoria y rendimiento
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import SWINIR_WINDOW_SIZE, SWINIR_DEPTHS, SWINIR_CHANNELS, IMG_SIZE

def window_partition(x, window_size):
    """
    Divide una característica en ventanas no superpuestas
    Args:
        x: (B, H, W, C)
        window_size (int): Tamaño de ventana
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reconstruye características a partir de ventanas
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Tamaño de ventana
        H (int): Altura de la imagen
        W (int): Ancho de la imagen
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) por muestra cuando se aplica en el camino principal de bloques residuales.
    
    Esta técnica mejora la regularización al aleatorizar qué capas se activan durante el entrenamiento.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Trabajar con tensores de diferentes dimensiones, no solo 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarizar
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    """
    MLP mejorado con inicialización optimizada y función de activación GELU.
    
    Esta implementación utiliza dos capas lineales con dropout y
    inicialización cuidadosa para mejor convergencia.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización mejorada
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ProgressiveWindowAttention(nn.Module):
    """
    Atención de ventana progresiva que captura contexto a múltiples escalas.
    
    Esta implementación extiende la atención de ventana estándar con la capacidad de 
    atender a ventanas de diferentes tamaños, lo que permite capturar dependencias 
    de largo alcance de manera más eficiente.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Proyecciones para Q, K, V con inicialización mejorada
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Parámetros para posición relativa
        # Límites para la coordenada de la posición relativa
        self.window_size = window_size
        self.num_relative_distance = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )
        
        # Generar pares de coordenadas para cada posición en la ventana
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # Calcular posición relativa de cada par de coordenadas
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # desplazamiento del origen
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        # Índice de mapa de coordenadas relativas a tabla de sesgos
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Inicialización de pesos
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        nn.init.zeros_(self.proj.bias)
        
        # Mecanismo de atención multi-escala 
        self.multi_scale_heads = min(2, num_heads // 2)  # Usar hasta 2 cabezas para multi-escala
        if self.multi_scale_heads > 0:
            self.scale_shift = nn.Parameter(torch.zeros(2))  # Parámetros aprendibles para balancear escalas

    def forward(self, x, mask=None):
        """
        Args:
            x: entrada con forma [B_, N, C]
            mask: máscara con forma [nW, Mh*Mw, Mh*Mw] o None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]
        
        # Factor de escala para producir softmax más estable
        q = q * self.scale
        
        # Cálculo de atención estándar
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]
        
        # Cálculo y adición de bias de posición relativa
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Aplicar softmax y dropout en la matriz de atención
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Implementación de atención multi-escala para las primeras multi_scale_heads cabezas
        if self.multi_scale_heads > 0 and self.training:
            # Usar las primeras self.multi_scale_heads cabezas para atención multi-escala
            attn_multi = attn[:, :self.multi_scale_heads].clone()
            
            # Modificación de la atención a escala gruesa
            window_size_coarse = self.window_size // 2
            if window_size_coarse >= 2:  # Solo si la ventana es lo suficientemente grande
                # Promediar atención sobre bloques 2x2 para simular escala gruesa
                attn_coarse = attn_multi.view(B_, self.multi_scale_heads, 
                                             self.window_size, self.window_size,
                                             self.window_size, self.window_size)
                attn_coarse = attn_coarse.permute(0, 1, 2, 4, 3, 5).contiguous()
                attn_coarse = attn_coarse.view(B_, self.multi_scale_heads, 
                                              self.window_size * self.window_size,
                                              self.window_size * self.window_size)
                
                # Combinar ambas escalas con parámetros aprendibles
                scale_weight = F.softmax(self.scale_shift, dim=0)
                attn[:, :self.multi_scale_heads] = scale_weight[0] * attn_multi + scale_weight[1] * attn_coarse
        
        # Aplicar la atención para obtener el valor de salida
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Proyección final
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Bloque transformador Swin con ventanas de atención y MLP.
    
    Esta implementación añade normalización de capas mejorada, stochastic depth adaptativo
    y mejor regularización.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, dropout=0., attn_dropout=0., drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # Si el tamaño de la imagen es menor que el tamaño de ventana, no usar shift
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            
        assert 0 <= self.shift_size < self.window_size, "shift_size must be between 0 and window_size"
        
        # Normalización de layer antes de la atención de ventana
        self.norm1 = norm_layer(dim)
        
        # Atención de ventana mejorada con soporte progresivo
        self.attn = ProgressiveWindowAttention(
            dim, window_size=window_size, num_heads=num_heads, 
            qkv_bias=qkv_bias, dropout=attn_dropout
        )
        
        # Stochastic depth regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Normalización y MLP después de la atención
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, dropout=dropout)
        
        # Calcular la máscara de atención para shifted windows
        if self.shift_size > 0:
            # Calcular máscara de atención para SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                    
            # Particionar la máscara en ventanas
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        # Skip connection 1
        shortcut = x
        
        # Normalización, reshape para ventanas y atención
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Ciclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Particionamiento de ventanas
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # Atención de ventana
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
            
        # Flatten and add first skip connection
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # MLP with skip connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class AdaptiveResidualBlock(nn.Module):
    """
    Bloque residual adaptativo que combina convoluciones y atención 
    para capturar dependencias espaciales y de canal.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
        # Atención de canal
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Atención espacial
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Parámetro aprendible para balancear skip connection
        self.beta = nn.Parameter(torch.zeros(1))
        
        # Inicialización optimizada
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='linear')
        
        # Aplicar inicialización a todas las convoluciones en atención
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.spatial_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='sigmoid')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        identity = x
        
        # Flujo de convolución
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        
        # Aplicar atención de canal
        ca = self.channel_attention(out)
        out = out * ca
        
        # Aplicar atención espacial
        sa = self.spatial_attention(out)
        out = out * sa
        
        # Activación y conexión residual modulada
        out = self.act(out)
        out = identity + self.beta * out
        
        return out

class ResidualSwinTransformerBlock(nn.Module):
    """
    Bloque residual que combina Swin Transformer con convoluciones 
    y atención mejorada para procesamiento de imágenes médicas.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, dropout=0., attn_dropout=0.):
        super().__init__()
        
        # Bloques transformadores con shift-window
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                input_resolution=input_resolution,
                num_heads=num_heads, 
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias,
                dropout=dropout, 
                attn_dropout=attn_dropout,
                drop_path=0.1 * (i / max(1, depth-1))  # Aumento progresivo de drop_path
            )
            for i in range(depth)
        ])
        
        # Capa adaptativa para mejorar la consistencia entre dominios
        self.adaptive_block = AdaptiveResidualBlock(dim)
        
        # Normalización final para estabilidad
        self.norm = nn.LayerNorm(dim)
        
        # Parámetro para balancear la conexión residual principal
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Filtro de suavizado para preservar detalles de bajo nivel
        self.smooth_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Inicialización
        nn.init.constant_(self.gamma, 1.0)
        nn.init.kaiming_normal_(self.smooth_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Guardar entrada original
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        original = x
        
        # Pasar por bloques transformadores
        for blk in self.blocks:
            x = blk(x)
        
        # Normalización para estabilidad 
        x = self.norm(x)
        
        # Transformar a formato de imagen
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Aplicar procesamiento convolucional adaptativo
        x = self.adaptive_block(x)
        
        # Aplicar filtro de suavizado para preservar detalles
        x = self.smooth_conv(x)
        
        # Volver al formato de secuencia
        x = x.flatten(2).transpose(1, 2)
        
        # Combinar con la entrada original
        return original + self.gamma * x

class GeneratorSwinIR(nn.Module):
    """
    Generador mejorado basado en SwinIR optimizado para imágenes confocales.
    
    Incorpora múltiples bloques residuales basados en Swin Transformer,
    mecanismos de atención adaptativos y conexiones residuales optimizadas.
    """
    def __init__(self, img_size=256, in_channels=3, embed_dim=128, depths=[4,4,4], 
                 num_heads=8, window_size=16, use_checkpoint=False):
        super().__init__()
        
        # Asegurar dimensiones correctas
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        
        # Verificar divisibilidad por window_size
        for size in img_size:
            assert size % window_size == 0, f"El tamaño de imagen {size} debe ser divisible por window_size {window_size}"
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        
        # Primera capa convolucional para extracción de características
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # Múltiples bloques residuales Swin Transformer
        self.rstb_blocks = nn.ModuleList()
        for depth in depths:
            self.rstb_blocks.append(
                ResidualSwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=img_size,
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    dropout=0.1,
                    attn_dropout=0.1
                )
            )
        
        # Capa de normalización antes de la reconstrucción
        self.norm = nn.LayerNorm(embed_dim)
        
        # Convolución final para reconstrucción
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // 2, in_channels, kernel_size=3, padding=1)
        )
        
        # Parámetro para controlar la conexión residual global
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Normalización adaptativa para mejorar la calidad de la imagen
        self.adaptive_norm = nn.InstanceNorm2d(in_channels, affine=True)
        
        # Inicialización de pesos para mejor convergencia
        self._init_weights()
        
    def _init_weights(self):
        """Inicialización optimizada de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def _check_image_size(self, x):
        """Garantiza que las dimensiones sean múltiplos del tamaño de ventana"""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, mod_pad_h, mod_pad_w

    def forward(self, x):
        """
        Forward pass del generador
        Args:
            x: Tensor de entrada de forma (B, C, H, W)
        Returns:
            output: Tensor de salida de forma (B, C, H, W)
        """
        # Guardar la entrada para la conexión residual
        identity = x
        
        # Verificar y ajustar tamaño de imagen si es necesario
        x, mod_pad_h, mod_pad_w = self._check_image_size(x)
        
        # Extracción de características inicial
        feat = self.conv_first(x)
        B, C, H, W = feat.shape
        
        # Transformar a secuencia para bloques Transformer
        x = feat.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Pasar por bloques residuales SwinIR
        for block in self.rstb_blocks:
            x = block(x)
        
        # Normalización antes de la reconstrucción
        x = self.norm(x)
        
        # Transformar de vuelta a formato de imagen
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # Reconstrucción final
        out = self.conv_last(x)
        
        # Eliminar padding si se agregó
        if mod_pad_h > 0 or mod_pad_w > 0:
            out = out[:, :, :H-mod_pad_h, :W-mod_pad_w]
        
        # Combinar con la entrada original usando conexión residual escalada
        output = identity + self.gamma * self.adaptive_norm(out)
        
        return output

if __name__ == "__main__":
    # Código de prueba para verificar la implementación
    torch.manual_seed(42)
    
    # Definir modelo SwinIR con parámetros optimizados
    model = GeneratorSwinIR(
        img_size=256,
        in_channels=3,
        embed_dim=128,
        depths=[4, 4, 4],
        num_heads=8,
        window_size=16
    )
    
    # Contar y mostrar número total de parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params/1e6:.2f}M")
    print(f"Parámetros entrenables: {trainable_params/1e6:.2f}M")
    
    # Generar una entrada de prueba
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Medir tiempo de inferencia
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        output = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    print(f"Forma de salida: {output.shape}")
    print(f"Rango de salida: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Tiempo de inferencia: {(end-start)*1000:.2f}ms")