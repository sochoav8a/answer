# models/cyclegan_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import itertools
import random

from models.generator_swinir import GeneratorSwinIR
from models.resnet_generator import ResnetGenerator
from models.discriminator import NLayerDiscriminator
from utils.losses import VGGFeatureExtractor, PerceptualLoss

class ImagePool:
    """
    Buffer de imágenes históricas para discriminadores.
    
    Este buffer mantiene un historial de imágenes generadas previamente,
    lo que ayuda a reducir la oscilación durante el entrenamiento del discriminador.
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        """
        Consulta el buffer con las imágenes actuales y devuelve imágenes
        que pueden ser una mezcla de históricas y actuales.
        
        Args:
            images (torch.Tensor): Imágenes generadas actuales
            
        Returns:
            torch.Tensor: Imágenes para entrenar el discriminador
        """
        if self.pool_size == 0:
            return images
            
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                # Si el buffer no está lleno, almacena la imagen y devuélvela
                self.num_imgs += 1
                self.images.append(image.clone())
                return_images.append(image)
            else:
                # Si el buffer está lleno, con probabilidad 0.5:
                # - Devuelve una imagen aleatoria del buffer y reemplázala con la actual
                # - O simplemente devuelve la imagen actual
                if random.uniform(0, 1) > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    temp = self.images[random_id].clone()
                    self.images[random_id] = image.clone()
                    return_images.append(temp)
                else:
                    return_images.append(image)
                    
        return torch.cat(return_images, 0)

class CycleGANModel(nn.Module):
    """
    Implementación mejorada del modelo CycleGAN para la mejora de imágenes confocales,
    con generador SwinIR para la transformación LR→HR y generador ResNet para HR→LR.
    
    Características clave:
    - Escaladores separados para generadores y discriminadores
    - Buffer para imágenes históricas (estabilidad del discriminador)
    - Regularización R1 para discriminadores
    - Sistema de pérdidas optimizado con pesos adaptativos
    - Soporte completo para entrenamiento con precisión mixta
    - Normalización optimizada para características VGG
    """
    def __init__(self, config):
        super(CycleGANModel, self).__init__()
        self.config = config  
        self.device = config.DEVICE
        self.lambda_cycle = config.LAMBDA_CYCLE
        self.lambda_idt = config.LAMBDA_IDENTITY
        self.lambda_perceptual = config.LAMBDA_PERCEPTUAL
        self.total_epochs = config.EPOCHS
        self.current_epoch = 0
        
        # Escaladores separados para generadores y discriminadores
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()
        
        # Parámetros de regularización
        self.r1_gamma = 0.0  # Regularización R1 para discriminadores
        
        # Extractor VGG para pérdida perceptual
        self.vgg_extractor = VGGFeatureExtractor(
            feature_layer=35,  # Capa de características VGG
            use_bn=False,      # No usar batch norm en VGG
            vgg_normalize=True, # Normalizar según ImageNet
            requires_grad=False # No entrenar VGG
        )
        self.perceptual_loss_fn = PerceptualLoss(self.vgg_extractor)
        
        # Generadores
        # G: LR → HR (SwinIR para alta calidad)
        self.netG = GeneratorSwinIR().to(self.device)
        # F: HR → LR (ResNet más ligero)
        self.netF = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(self.device)

        # Discriminadores
        # D_HR: Discrimina entre HR real y generada
        self.netD_HR = NLayerDiscriminator(input_nc=3).to(self.device)
        # D_LR: Discrimina entre LR real y generada
        self.netD_LR = NLayerDiscriminator(input_nc=3).to(self.device)

        # Criterios de pérdida
        self.criterionGAN = nn.MSELoss()  # Pérdida adversarial
        self.criterionCycle = nn.L1Loss()  # Pérdida de ciclo
        self.criterionIdt = nn.L1Loss()    # Pérdida de identidad
        
        # Inicialización de pesos optimizada
        self._init_weights()

        # Buffer de imágenes históricas para estabilidad
        self.fake_HR_pool = ImagePool(pool_size=50)
        self.fake_LR_pool = ImagePool(pool_size=50)

        # Optimizadores separados
        self.optimizer_G = optim.Adam(
            itertools.chain(self.netG.parameters(), self.netF.parameters()),
            lr=config.LR_GENERATOR, 
            betas=(config.BETA1, config.BETA2),
            weight_decay=1e-5  # Regularización adicional
        )
        
        self.optimizer_D = optim.Adam(
            itertools.chain(self.netD_HR.parameters(), self.netD_LR.parameters()),
            lr=config.LR_DISCRIMINATOR, 
            betas=(config.BETA1, config.BETA2),
            weight_decay=1e-5
        )
        
        # Diccionario para almacenar pérdidas individuales para monitoreo
        self.loss_dict = {}
        
    def _init_weights(self):
        """
        Inicialización especializada de pesos para componentes críticos.
        """
        # Inicialización específica para discriminadores
        for m in itertools.chain(self.netD_HR.modules(), self.netD_LR.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def set_input(self, input_dict):
        """
        Configura las entradas del modelo a partir del lote de datos.
        
        Args:
            input_dict (dict): Diccionario con claves 'LR' y 'HR'
        """
        self.real_LR = input_dict['LR'].to(self.device)
        self.real_HR = input_dict['HR'].to(self.device)
        
        # Almacenar dimensiones para posible recorte o padding
        self.B, self.C, self.H, self.W = self.real_LR.shape

    def forward(self):
        """
        Ejecución hacia adelante del modelo completo (no se usa en entrenamiento,
        sino que se divide en pasos de optimización separados).
        """
        with autocast():
            # G: LR → HR (upscale/mejora)
            self.fake_HR = self.netG(self.real_LR)
            
            # F: Fake HR → LR (reconstrucción para ciclo)
            self.rec_LR = self.netF(self.fake_HR)

            # F: HR → LR (downscale)
            self.fake_LR = self.netF(self.real_HR)
            
            # G: Fake LR → HR (reconstrucción para ciclo)
            self.rec_HR = self.netG(self.fake_LR)

            # Mapeo de identidad: preservar contenido en dominio destino
            # G(HR) ≈ HR para preservar detalles de alta resolución
            self.idt_HR = self.netG(self.real_HR)
            
            # F(LR) ≈ LR para preservar características LR
            self.idt_LR = self.netF(self.real_LR)

    def backward_G(self):
        """
        Cálculo de pérdidas y retropropagación para los generadores G y F.
        """
        with autocast():
            # 1) Pérdidas adversariales
            # G debe engañar a D_HR generando HR creíbles
            pred_fake_HR = self.netD_HR(self.fake_HR)
            loss_G_adv = self.criterionGAN(
                pred_fake_HR, 
                torch.ones_like(pred_fake_HR, device=self.device)
            )
            
            # F debe engañar a D_LR generando LR creíbles
            pred_fake_LR = self.netD_LR(self.fake_LR)
            loss_F_adv = self.criterionGAN(
                pred_fake_LR, 
                torch.ones_like(pred_fake_LR, device=self.device)
            )
            
            # 2) Pérdidas de ciclo (consistencia)
            # LR → HR → LR debe preservar la imagen original
            loss_cycle_LR = self.criterionCycle(self.rec_LR, self.real_LR)
            
            # HR → LR → HR debe preservar la imagen original
            loss_cycle_HR = self.criterionCycle(self.rec_HR, self.real_HR)
            
            # Pérdida de ciclo combinada
            loss_cycle = loss_cycle_LR + loss_cycle_HR

            # 3) Pérdidas de identidad (preservación)
            # G(HR) debe ser similar a HR
            loss_idt_HR = self.criterionIdt(self.idt_HR, self.real_HR)
            
            # F(LR) debe ser similar a LR
            loss_idt_LR = self.criterionIdt(self.idt_LR, self.real_LR)
            
            # Pérdida de identidad combinada
            loss_idt = loss_idt_HR + loss_idt_LR

            # 4) Pérdida perceptual (características VGG)
            # Normalización y preparación para VGG
            rec_LR_vgg = (self.rec_LR + 1) / 2  # [-1,1] → [0,1]
            real_LR_vgg = (self.real_LR + 1) / 2
            
            # Asegurar 3 canales para entrada a VGG (si es grayscale)
            if rec_LR_vgg.size(1) == 1:
                rec_LR_vgg = rec_LR_vgg.repeat(1, 3, 1, 1)
            if real_LR_vgg.size(1) == 1:
                real_LR_vgg = real_LR_vgg.repeat(1, 3, 1, 1)
            
            # Extraer características VGG de las imágenes reales sin gradientes
            with torch.no_grad():
                real_features = self.perceptual_loss_fn.feature_extractor(real_LR_vgg)
                
            # Extraer características VGG de las imágenes generadas
            fake_features = self.perceptual_loss_fn.feature_extractor(rec_LR_vgg)
            
            # Calcular pérdida perceptual (diferencia de características)
            loss_perceptual = self.perceptual_loss_fn.criterion(fake_features, real_features)
            
            # 5) Pérdida total para generadores con pesos configurables
            self.loss_G = (
                loss_G_adv + loss_F_adv +
                self.lambda_cycle * loss_cycle +
                self.lambda_idt * loss_idt +
                self.lambda_perceptual * loss_perceptual
            )
            
            # Almacenar pérdidas individuales para monitoreo
            self.loss_dict.update({
                'G_adv': loss_G_adv.item(),
                'F_adv': loss_F_adv.item(),
                'cycle': loss_cycle.item(),
                'identity': loss_idt.item(),
                'perceptual': loss_perceptual.item()
            })

        # Backpropagation con escalador de precisión mixta
        self.scaler_G.scale(self.loss_G).backward()

    def backward_D_basic(self, netD, real, fake):
        """
        Cálculo de pérdida básica para un discriminador.
        
        Args:
            netD: Discriminador a entrenar
            real: Imágenes reales
            fake: Imágenes generadas
            
        Returns:
            loss_D: Pérdida total para este discriminador
        """
        with autocast():
            # Pérdida para imágenes reales (deben clasificarse como reales)
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(
                pred_real, 
                torch.ones_like(pred_real, device=self.device)
            )
            
            # Pérdida para imágenes falsas (deben clasificarse como falsas)
            pred_fake = netD(fake.detach())  # Detach para no propagar gradientes al generador
            loss_D_fake = self.criterionGAN(
                pred_fake, 
                torch.zeros_like(pred_fake, device=self.device)
            )
            
            # Regularización R1 (opcional)
            loss_D_reg = 0.0
            if self.r1_gamma > 0:
                # Calcular gradiente con respecto a las entradas reales
                real.requires_grad_(True)
                pred_real_reg = netD(real)
                grad_real = torch.autograd.grad(
                    outputs=pred_real_reg.sum(),
                    inputs=real,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Penalización del gradiente (regularización R1)
                loss_D_reg = self.r1_gamma * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                real.requires_grad_(False)
            
            # Combinar pérdidas
            loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_reg
            
            return loss_D

    def backward_D(self):
        """
        Cálculo de pérdidas y retropropagación para ambos discriminadores.
        """
        with autocast():
            # 1) Discriminador HR (distingue HR real vs generada)
            # Usar buffer de imágenes para mayor estabilidad
            fake_HR = self.fake_HR_pool.query(self.fake_HR)
            self.loss_D_HR = self.backward_D_basic(self.netD_HR, self.real_HR, fake_HR)
            
            # 2) Discriminador LR (distingue LR real vs generada)
            fake_LR = self.fake_LR_pool.query(self.fake_LR)
            self.loss_D_LR = self.backward_D_basic(self.netD_LR, self.real_LR, fake_LR)
            
            # Pérdida total para ambos discriminadores
            self.loss_D = self.loss_D_HR + self.loss_D_LR
            
            # Almacenar pérdidas individuales para monitoreo
            self.loss_dict.update({
                'D_HR': self.loss_D_HR.item(),
                'D_LR': self.loss_D_LR.item()
            })

        # Backpropagation con escalador de precisión mixta
        self.scaler_D.scale(self.loss_D).backward()

    def optimize_parameters(self):
        """
        Realiza un paso completo de optimización (forward + backward + optimize).
        """
        # Forward pass
        self.forward()
        
        # 1) Actualizar generadores G y F
        # Desactivar la optimización de discriminadores
        self.set_requires_grad([self.netD_HR, self.netD_LR], False)
        
        # Limpiar gradientes y calcular pérdidas para generadores
        self.optimizer_G.zero_grad(set_to_none=True)  # Más eficiente
        self.backward_G()
        
        # Paso del optimizador con escalador
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

        # 2) Actualizar discriminadores D_HR y D_LR
        # Reactivar la optimización de discriminadores
        self.set_requires_grad([self.netD_HR, self.netD_LR], True)
        
        # Limpiar gradientes y calcular pérdidas para discriminadores
        self.optimizer_D.zero_grad(set_to_none=True)
        self.backward_D()
        
        # Paso del optimizador con escalador
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Establece el valor de requires_grad para todos los parámetros de las redes.
        
        Args:
            nets (list): Lista de redes
            requires_grad (bool): Si los parámetros requieren gradiente
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """
        Actualiza las tasas de aprendizaje usando un esquema de decaimiento.
        
        Implementa un esquema de decaimiento lineal en la segunda mitad del entrenamiento
        para estabilizar al final.
        """
        self.current_epoch += 1
        
        # Esquema de decaimiento lineal en la segunda mitad
        half_epoch = self.total_epochs // 2
        if self.current_epoch > half_epoch:
            # Decaimiento lineal desde lr_inicial hasta lr_final
            decay_factor = max(0, (self.total_epochs - self.current_epoch) / 
                                (self.total_epochs - half_epoch))
            
            # Factor mínimo de decaimiento (no ir por debajo del 10% de la lr inicial)
            decay_factor = max(0.1, decay_factor)
        else:
            # Mantener tasa de aprendizaje original en la primera mitad
            decay_factor = 1.0

        # Calcular nuevas tasas de aprendizaje
        new_lr_g = self.config.LR_GENERATOR * decay_factor
        new_lr_d = self.config.LR_DISCRIMINATOR * decay_factor

        # Actualizar optimizadores
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_g
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_d
            
        # Imprimir información de actualización
        if self.current_epoch % 10 == 0 or self.current_epoch == 1:
            print(f"Época {self.current_epoch}: LR_G = {new_lr_g:.6f}, LR_D = {new_lr_d:.6f}")

    def get_current_losses(self):
        """
        Devuelve un diccionario con todas las pérdidas actuales.
        
        Returns:
            dict: Diccionario con todas las pérdidas
        """
        losses = {
            'loss_G': self.loss_G.item() if hasattr(self, 'loss_G') else 0.0,
            'loss_D': self.loss_D.item() if hasattr(self, 'loss_D') else 0.0
        }
        # Añadir pérdidas individuales si están disponibles
        if hasattr(self, 'loss_dict') and self.loss_dict:
            losses.update(self.loss_dict)
            
        return losses
        
    def get_current_visuals(self):
        """
        Devuelve un diccionario con las imágenes actuales para visualización.
        
        Returns:
            dict: Diccionario con tensores de imágenes desnormalizadas
        """
        # Desnormalizar imágenes de [-1,1] a [0,1]
        def denormalize(img):
            return (img * 0.5) + 0.5
            
        return {
            'real_LR': denormalize(self.real_LR),
            'fake_HR': denormalize(self.fake_HR),
            'rec_LR': denormalize(self.rec_LR),
            'real_HR': denormalize(self.real_HR),
            'fake_LR': denormalize(self.fake_LR),
            'rec_HR': denormalize(self.rec_HR)
        }
        
    def save_networks(self, epoch, checkpoint_dir, best=False):
        """
        Guarda el estado de todas las redes y optimizadores.
        
        Args:
            epoch (int): Época actual
            checkpoint_dir (str): Directorio para guardar checkpoints
            best (bool): Si es el mejor modelo hasta ahora
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        if best:
            checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            
        torch.save({
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netF_state_dict': self.netF.state_dict(),
            'netD_HR_state_dict': self.netD_HR.state_dict(),
            'netD_LR_state_dict': self.netD_LR.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scaler_G_state_dict': self.scaler_G.state_dict(),
            'scaler_D_state_dict': self.scaler_D.state_dict(),
        }, checkpoint_path)
        
        print(f"Red guardada en {checkpoint_path}")
        
    def load_networks(self, checkpoint_path):
        """
        Carga el estado de todas las redes y optimizadores desde un checkpoint.
        
        Args:
            checkpoint_path (str): Ruta al archivo de checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Cargar estado de los generadores
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netF.load_state_dict(checkpoint['netF_state_dict'])
        
        # Cargar estado de los discriminadores
        self.netD_HR.load_state_dict(checkpoint['netD_HR_state_dict'])
        self.netD_LR.load_state_dict(checkpoint['netD_LR_state_dict'])
        
        # Cargar estado de los optimizadores
        if 'optimizer_G_state_dict' in checkpoint:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            
        # Cargar estado de los escaladores si existen
        if 'scaler_G_state_dict' in checkpoint:
            self.scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
            self.scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
            
        # Actualizar época actual
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
            
        print(f"Checkpoint cargado desde {checkpoint_path} (época {self.current_epoch})")