# utils/losses.py

import torch
import torch.nn as nn
from torchvision.models import vgg19, vgg19_bn
from torchvision.models import VGG19_Weights, VGG19_BN_Weights

class VGGFeatureExtractor(nn.Module):
    """
    Extrae características de una imagen utilizando un modelo VGG19 pre-entrenado.
    
    Parámetros:
        feature_layer (int): Índice de la capa hasta la cual se extraen las características.
        use_bn (bool): Si se debe utilizar la versión con BatchNorm de VGG19.
        vgg_normalize (bool): Si se normaliza la imagen de entrada con medias/std de ImageNet.
        requires_grad (bool): Si se actualizarán los pesos del extractor (por defecto, False).
    """
    def __init__(self, feature_layer=35, use_bn=False, vgg_normalize=True, requires_grad=False):
        super(VGGFeatureExtractor, self).__init__()
        # En torchvision >=0.13, se usa 'weights=' en lugar de 'pretrained='
        if use_bn:
            backbone = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
        else:
            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Extrae las capas hasta la capa indicada (inclusive)
        self.features = nn.Sequential(*list(backbone.features.children())[:feature_layer+1])
        
        self.vgg_normalize = vgg_normalize
        if self.vgg_normalize:
            # Normalización según ImageNet
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Calcula las características de la imagen.
        
        Se asume que 'x' está en el rango [0, 1], si vgg_normalize=True.
        """
        if self.vgg_normalize:
            x = (x - self.mean) / self.std
        
        out = self.features(x)
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, criterion=nn.L1Loss()):
        """
        Inicializa la pérdida perceptual usando un extractor de características pre-entrenado.
        """
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.criterion = criterion
        self.feature_extractor.eval()

    def forward(self, input, target):
        """
        Calcula la pérdida perceptual entre input y target.
        Ambos tensores deben tener shape (B, 3, H, W) y rango ~[0,1].
        """
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        input_features = self.feature_extractor(input)
        loss = self.criterion(input_features, target_features)
        return loss
