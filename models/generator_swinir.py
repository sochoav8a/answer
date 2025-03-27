"""
models/generator_swinir.py

Implementación mejorada del generador SwinIR con:
- Mayor capacidad de representación
- Mecanismos de atención mejorados
- Conexiones residuales optimizadas
- Regularización mejorada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SWINIR_WINDOW_SIZE, SWINIR_DEPTHS, SWINIR_CHANNELS, IMG_SIZE

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Añadimos la clase DropPath que faltaba
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
        # Inicialización mejorada
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Matrices de parámetros mejoradas
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Mecanismo de atención cruzada
        self.cross_scale_attn = nn.MultiheadAttention(dim, num_heads=2, dropout=dropout)
        
        # Inicialización mejorada
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.normal_(self.qkv.bias, std=1e-6)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, std=1e-6)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Atención multi-escala
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Conexión cruzada
        x = x + self.cross_scale_attn(x, x, x)[0]
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)
        
        # Regularización mejorada
        self.stochastic_depth = torch.rand(1).item() < drop_path

    def forward(self, x):
        if self.training and self.stochastic_depth:
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ResidualSwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, dropout=0., attn_dropout=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 dropout=dropout, attn_dropout=attn_dropout,
                                 drop_path=0.1 * (i / depth))  # Aumento progresivo de drop_path
            for i in range(depth)
        ])
        
        # Capa convolucional mejorada
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # Depthwise convolution
            nn.Conv2d(dim, dim, 1),  # Pointwise convolution
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # Mecanismo de atención de canal
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        
        for blk in self.blocks:
            x = blk(x)
            
        # Transformación mejorada
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x) * self.channel_attn(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class GeneratorSwinIR(nn.Module):
    def __init__(self, img_size=256, in_channels=3, embed_dim=128, depths=[4,4,4], num_heads=8, window_size=16):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.rstb_blocks = nn.ModuleList()
        
        # Asegurar divisibilidad
        assert img_size % window_size == 0, "El tamaño de imagen debe ser divisible por window_size"
        
        for depth in depths:
            self.rstb_blocks.append(
                ResidualSwinTransformerBlock(
                    embed_dim,
                    (img_size, img_size),
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_size)
            )
        
        # Capa final mejorada
        self.conv_last = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(embed_dim, in_channels, 3, padding=1)
        )
        
        # Conexión residual aprendible
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        feat = self.conv_first(x)
        B, C, H, W = feat.shape
        
        x = feat.flatten(2).transpose(1, 2)
        for block in self.rstb_blocks:
            x = block(x)
            
        x = x.transpose(1, 2).view(B, C, H, W)
        out = self.conv_last(x)
        return identity + self.gamma * out  # Conexión residual escalada

if __name__ == "__main__":
    # Prueba mejorada
    model = GeneratorSwinIR(
        img_size=256,
        embed_dim=128,
        depths=[4,4,4],
        num_heads=8,
        window_size=16
    )
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Forma de salida: {output.shape}")
    print(f"Rango de salida: [{output.min():.3f}, {output.max():.3f}]")