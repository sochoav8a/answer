"""
models/resnet_generator.py

Implementación de un generador basado en una arquitectura ResNet para la transformación
de imágenes HR a LR en el esquema CycleGAN. La arquitectura mantiene la resolución (256x256)
y está compuesta por:
  - Una capa inicial con padding reflectivo, seguida de convolución, normalización y activación.
  - Una serie de bloques residuales (ResnetBlock) que aprenden transformaciones refinadas.
  - Una capa final que reconstruye la imagen de salida.
  
Esta arquitectura se inspira en el generador usado en CycleGAN y otras redes de traducción de imagen.
"""

import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    """
    Define un bloque residual con dos capas convolucionales, utilizando padding reflectivo,
    normalización de instancia y activación ReLU. La conexión residual suma la entrada al
    resultado de las operaciones convolucionales.
    """
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        # Primera capa: padding, convolución, normalización y activación.
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f"Padding [{padding_type}] no implementado")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # Segunda capa: padding, convolución y normalización.
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        # Se suma la entrada (skip connection) al resultado del bloque convolucional.
        out = x + self.conv_block(x)
        return out

class ResnetGenerator(nn.Module):
    """
    Generador basado en arquitectura ResNet para la transformación de imágenes.
    Se utiliza para la dirección HR -> LR en el esquema CycleGAN.
    
    La arquitectura se compone de:
      - Una capa inicial que extrae características de la imagen de entrada.
      - Una serie de bloques residuales (ResnetBlock) que refinan la representación.
      - Una capa final que reconstruye la imagen de salida, seguida de activación Tanh.
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """
        Parámetros:
            input_nc (int): Número de canales de la imagen de entrada.
            output_nc (int): Número de canales de la imagen de salida.
            ngf (int): Número de filtros en la capa inicial.
            n_blocks (int): Número de bloques residuales.
            padding_type (str): Tipo de padding ('reflect', 'replicate' o 'zero').
            norm_layer (callable): Capa de normalización a utilizar.
            use_dropout (bool): Indica si se usa dropout.
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d)

        model = []
        # Capa inicial: Padding reflectivo, convolución, normalización y activación.
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        
        # No se realiza downsampling ya que se desea mantener la resolución 256x256.
        # Se agregan n_blocks bloques residuales.
        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        # Capa final: Reconstrucción de la imagen con activación Tanh.
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        """
        Parámetro:
            input (Tensor): Imagen de entrada (B, input_nc, H, W)
        Retorna:
            Tensor: Imagen de salida transformada (B, output_nc, H, W)
        """
        return self.model(input)

if __name__ == "__main__":
    # Prueba rápida del generador ResNet
    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    print(model)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)
