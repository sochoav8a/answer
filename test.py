import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from data.dataset import ConfocalDataset, get_validation_transforms
from models.cyclegan_model import CycleGANModel
from utils.metrics import calculate_psnr, calculate_ssim

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.netG.load_state_dict(checkpoint['netG_state_dict'])
    model.netF.load_state_dict(checkpoint['netF_state_dict'])
    model.netD_HR.load_state_dict(checkpoint['netD_HR_state_dict'])
    model.netD_LR.load_state_dict(checkpoint['netD_LR_state_dict'])
    print(f"Checkpoint cargado: {checkpoint_path}")

def find_latest_checkpoint(checkpoints_dir, model_name):
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith(model_name) and f.endswith('.pth')]
    if not checkpoints:
        return None
    epochs = [int(f.split('_epoch_')[-1].split('.pth')[0]) for f in checkpoints]
    max_epoch = max(epochs)
    latest_checkpoint = f"{model_name}_epoch_{max_epoch}.pth"
    return os.path.join(checkpoints_dir, latest_checkpoint)

def main(args):
    config.setup_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"Inferencia en dispositivo: {device}")

    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)

    transform = get_validation_transforms(config.IMG_SIZE)

    if os.path.exists(config.VALID_PATH):
        print("Usando dataset de validación pareado.")
        dataset = ConfocalDataset(
            lr_dir=config.VALID_PATH,
            transform_lr=transform,
            transform_hr=transform,
            paired=True
        )
        use_ground_truth = True
    else:
        print("No se encontró archivo de validación. Abortando.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    model = CycleGANModel(config).to(device)
    model.eval()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(config.CHECKPOINTS_DIR, config.MODEL_NAME)
    if checkpoint_path is None:
        print("No se encontró checkpoint. Abortando.")
        return

    load_checkpoint(model, checkpoint_path)

    total_psnr, total_ssim, count = 0.0, 0.0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            lr_image = data['LR'].to(device)
            fake_hr = model.netG(lr_image)

            fake_hr_denorm = (fake_hr * 0.5) + 0.5
            lr_image_denorm = (lr_image * 0.5) + 0.5

            result_path = os.path.join(config.RESULTS_DIR, f"fake_HR_idx_{data['LR_idx'].item():04d}.png")
            input_path = os.path.join(config.RESULTS_DIR, f"input_LR_idx_{data['LR_idx'].item():04d}.png")

            save_image(fake_hr_denorm, result_path, normalize=True)
            save_image(lr_image_denorm, input_path, normalize=True)

            if use_ground_truth:
                hr_image = data['HR'].to(device)
                hr_image_denorm = (hr_image * 0.5) + 0.5

                gt_path = os.path.join(config.RESULTS_DIR, f"gt_HR_idx_{data['HR_idx'].item():04d}.png")
                save_image(hr_image_denorm, gt_path, normalize=True)

                psnr = calculate_psnr(fake_hr_denorm, hr_image_denorm)
                ssim = calculate_ssim(fake_hr_denorm, hr_image_denorm)
                total_psnr += psnr
                total_ssim += ssim
                count += 1

                print(f"[{data['LR_idx'].item()}] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

    if use_ground_truth and count > 0:
        print(f"\nPromedio PSNR: {total_psnr/count:.4f}, Promedio SSIM: {total_ssim/count:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="Ruta al checkpoint.")
    args = parser.parse_args()
    main(args)