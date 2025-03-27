# models/cyclegan_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from models.generator_swinir import GeneratorSwinIR
from models.resnet_generator import ResnetGenerator
from models.discriminator import NLayerDiscriminator
from utils.losses import VGGFeatureExtractor, PerceptualLoss

class CycleGANModel(nn.Module):
    def __init__(self, config):
        super(CycleGANModel, self).__init__()
        self.config = config  
        self.device = config.DEVICE
        self.lambda_cycle = config.LAMBDA_CYCLE
        self.lambda_idt = config.LAMBDA_IDENTITY
        self.total_epochs = config.EPOCHS
        self.current_epoch = 0
        
        # AMP scaler
        self.scaler = GradScaler()

        # Perceptual Loss
        vgg_extractor = VGGFeatureExtractor(feature_layer=35, use_bn=False, vgg_normalize=True, requires_grad=False)
        self.perceptual_loss_fn = PerceptualLoss(vgg_extractor)

        # Generadores
        self.netG = GeneratorSwinIR().to(self.device)
        self.netF = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9).to(self.device)

        # Discriminadores
        self.netD_HR = NLayerDiscriminator(input_nc=3).to(self.device)
        self.netD_LR = NLayerDiscriminator(input_nc=3).to(self.device)

        # Pérdidas
        self.criterionGAN = nn.MSELoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.netG.parameters()) + list(self.netF.parameters()),
            lr=config.LR_GENERATOR, betas=(config.BETA1, config.BETA2)
        )
        self.optimizer_D = optim.Adam(
            list(self.netD_HR.parameters()) + list(self.netD_LR.parameters()),
            lr=config.LR_DISCRIMINATOR, betas=(config.BETA1, config.BETA2)
        )

    def set_input(self, input_dict):
        self.real_LR = input_dict['LR'].to(self.device)
        self.real_HR = input_dict['HR'].to(self.device)

    def forward(self):
        with autocast():
            self.fake_HR = self.netG(self.real_LR)
            self.rec_LR = self.netF(self.fake_HR)

            self.fake_LR = self.netF(self.real_HR)
            self.rec_HR = self.netG(self.fake_LR)

            self.idt_HR = self.netG(self.real_HR)
            self.idt_LR = self.netF(self.real_LR)

    def backward_G(self):
        with autocast():
            # 1) Adversarial
            pred_fake_HR = self.netD_HR(self.fake_HR)
            loss_G_adv = self.criterionGAN(pred_fake_HR, torch.ones_like(pred_fake_HR, device=self.device))

            pred_fake_LR = self.netD_LR(self.fake_LR)
            loss_F_adv = self.criterionGAN(pred_fake_LR, torch.ones_like(pred_fake_LR, device=self.device))

            # 2) Cycle loss
            loss_cycle_LR = self.criterionCycle(self.rec_LR, self.real_LR)
            loss_cycle_HR = self.criterionCycle(self.rec_HR, self.real_HR)
            loss_cycle = loss_cycle_LR + loss_cycle_HR

            # 3) Idt
            loss_idt_HR = self.criterionIdt(self.idt_HR, self.real_HR)
            loss_idt_LR = self.criterionIdt(self.idt_LR, self.real_LR)
            loss_idt = loss_idt_HR + loss_idt_LR

            # 4) Perceptual loss (VGG)
            rec_LR_vgg = (self.rec_LR + 1) / 2  # [-1,1] => [0,1]
            real_LR_vgg = (self.real_LR + 1) / 2

            # Asegurar 3 canales (para grayscale)
            if rec_LR_vgg.size(1) == 1:
                rec_LR_vgg = rec_LR_vgg.repeat(1, 3, 1, 1)
            if real_LR_vgg.size(1) == 1:
                real_LR_vgg = real_LR_vgg.repeat(1, 3, 1, 1)

            # Normalización VGG (¡CRUCIAL!)
            vgg_mean = torch.tensor([0.485, 0.456, 0.406], 
                                device=self.device).view(1, 3, 1, 1)
            vgg_std = torch.tensor([0.229, 0.224, 0.225], 
                                device=self.device).view(1, 3, 1, 1)

            rec_LR_vgg = (rec_LR_vgg - vgg_mean) / vgg_std
            real_LR_vgg = (real_LR_vgg - vgg_mean) / vgg_std

            # Detener gradientes en características reales
            with torch.no_grad():
                real_features = self.perceptual_loss_fn.feature_extractor(real_LR_vgg)
                
            fake_features = self.perceptual_loss_fn.feature_extractor(rec_LR_vgg)
            loss_perceptual = self.perceptual_loss_fn.criterion(fake_features, real_features)
            # Combinar
            self.loss_G = (
                loss_G_adv + loss_F_adv +
                self.config.LAMBDA_CYCLE * loss_cycle +
                self.config.LAMBDA_IDENTITY * loss_idt +
                loss_perceptual  # Pérdida perceptual
            )

        self.scaler.scale(self.loss_G).backward()

    def backward_D(self):
        with autocast():
            # Discriminador HR
            pred_real_HR = self.netD_HR(self.real_HR)
            loss_D_HR_real = self.criterionGAN(pred_real_HR, torch.ones_like(pred_real_HR, device=self.device))

            pred_fake_HR = self.netD_HR(self.fake_HR.detach())
            loss_D_HR_fake = self.criterionGAN(pred_fake_HR, torch.zeros_like(pred_fake_HR, device=self.device))
            loss_D_HR = 0.5 * (loss_D_HR_real + loss_D_HR_fake)

            # Discriminador LR
            pred_real_LR = self.netD_LR(self.real_LR)
            loss_D_LR_real = self.criterionGAN(pred_real_LR, torch.ones_like(pred_real_LR, device=self.device))

            pred_fake_LR = self.netD_LR(self.fake_LR.detach())
            loss_D_LR_fake = self.criterionGAN(pred_fake_LR, torch.zeros_like(pred_fake_LR, device=self.device))
            loss_D_LR = 0.5 * (loss_D_LR_real + loss_D_LR_fake)

            self.loss_D = loss_D_HR + loss_D_LR

        self.scaler.scale(self.loss_D).backward()

    def optimize_parameters(self):
        self.forward()

        # 1) G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.scaler.step(self.optimizer_G)

        # 2) D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.scaler.step(self.optimizer_D)

        # 3) update scaler
        self.scaler.update()

    def update_learning_rate(self):
        self.current_epoch += 1
        half_epoch = self.total_epochs // 2
        if self.current_epoch > half_epoch:
            decay_factor = (self.total_epochs - self.current_epoch) / (self.total_epochs - half_epoch)
        else:
            decay_factor = 1.0

        new_lr_g = self.config.LR_GENERATOR * decay_factor
        new_lr_d = self.config.LR_DISCRIMINATOR * decay_factor

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_g
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_d

    def get_current_losses(self):
        return {
            'loss_G': self.loss_G.item() if hasattr(self, 'loss_G') else 0.0,
            'loss_D': self.loss_D.item() if hasattr(self, 'loss_D') else 0.0
        }
