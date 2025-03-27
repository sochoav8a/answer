"""
models/discriminator.py

Implementación del discriminador tipo PatchGAN, utilizado en CycleGAN.
El discriminador toma una imagen de entrada y produce un mapa de decisiones 
(real/falso) por parches. Esto ayuda a capturar detalles locales y texturas.

La arquitectura se basa en:
  - Una serie de capas convolucionales con activación LeakyReLU.
  - Normalización de instancia para estabilizar el entrenamiento.
  - La capa final produce un mapa de salida (no se aplica activación, 
    ya que se usa en la función de pérdida adversarial).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class NLayerDiscriminator(nn.Module):
    """
    Discriminador tipo PatchGAN con normalización espectral.
    Se utiliza para diferenciar entre imágenes reales y generadas en el dominio correspondiente.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_spectral_norm=True):
        """
        Parámetros:
            input_nc (int): Número de canales de la imagen de entrada.
            ndf (int): Número de filtros en la primera capa.
            n_layers (int): Número de capas convolucionales intermedias.
            norm_layer (callable): Tipo de capa de normalización.
            use_spectral_norm (bool): Si se aplica normalización espectral.
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)
        
        # Primera capa: Convolución sin normalización pero con SN opcional
        if use_spectral_norm:
            sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
                        nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        # Capas intermedias: Se incrementa el número de filtros y se utiliza normalización.
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if use_spectral_norm:
                conv_block = [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                            kernel_size=4, stride=2, padding=1, bias=use_bias)),
                ]
                if norm_layer is not None:
                    conv_block.append(norm_layer(ndf * nf_mult))
                conv_block.append(nn.LeakyReLU(0.2, True))
                sequence.extend(conv_block)
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                              kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
        
        # Capa adicional con stride 1 para aumentar el tamaño del receptive field.
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if use_spectral_norm:
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                      kernel_size=4, stride=1, padding=1, bias=use_bias)),
            ]
            if norm_layer is not None:
                sequence.append(norm_layer(ndf * nf_mult))
            sequence.append(nn.LeakyReLU(0.2, True))
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                          kernel_size=4, stride=1, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Capa de salida: Produce un mapa de parches de decisión (real/falso).
        if use_spectral_norm:
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """
        Parámetro:
            input (Tensor): Imagen de entrada (B, input_nc, H, W)
        Retorna:
            Tensor: Mapa de salida del discriminador.
        """
        return self.model(input)