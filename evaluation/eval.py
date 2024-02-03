import sys
sys.path.append("/path/to/pipeline")

import torch
import torchvision.transforms.functional as F
from pytorch_fid import fid_score
from sklearn.metrics import mean_squared_error
from cycleGAN_model import CycleGAN
from generator import get_gen
from data_loader import CustomDataModule

# Load the trained model
model = CycleGAN(gen_name="unet", num_resblocks=6, hid_channels=64, optimizer=torch.optim.Adam, lr=2e-4, betas=(0.5, 0.999), lambda_idt=0.5, lambda_cycle=(10, 10), buffer_size=100, num_epochs=18, decay_epochs=1)
model.load_state_dict(torch.load("C:\Projects\git_cycleGAN\cycleGAN\model\cycle_gan_model.pth"))  # Replace with the actual path
model.eval()

# Load the data module
dm = CustomDataModule(monet_dir="/path/to/monet_dataset/*.jpg", photo_dir="/path/to/photo_dataset/*.jpg", loader_config={"num_workers": 4})
dm.setup("fit")

# Evaluation function
def evaluate_model(model, datamodule):
    # Initialize metrics
    fid_scores = []
    loss_D_fake_values = []
    loss_D_real_values = []
    loss_D_values = []
    loss_G_GAN_values = []
    loss_G_values = []
    loss_G_LI_values = []

    for batch in datamodule.val_dataloader():
        real_M = batch["monet"]
        real_P = batch["photo"]

        # Forward pass
        fake_M = model.gen_PM(real_P)
        fake_P = model.gen_MP(real_M)

        # Metrics computation
        fid_score_value = fid_score(F.to_tensor(real_M), F.to_tensor(fake_M))
        loss_D_fake_values.append(model.get_adv_loss(fake_M, model.disc_M).item())
        loss_D_real_values.append(model.get_adv_loss(real_M, model.disc_M).item())
        loss_D_values.append(model.get_disc_loss(real_M, fake_M, model.disc_M).item())
        loss_G_GAN_values.append(model.get_adv_loss(fake_M, model.disc_M).item())
        loss_G_values.append(model.get_gen_loss().item())
        loss_G_LI_values.append(mean_squared_error(real_M.cpu().numpy().flatten(), fake_M.cpu().numpy().flatten()))

        # Save FID score
        fid_scores.append(fid_score_value)

    # Calculate average scores
    avg_fid_score = sum(fid_scores) / len(fid_scores)
    avg_loss_D_fake = sum(loss_D_fake_values) / len(loss_D_fake_values)
    avg_loss_D_real = sum(loss_D_real_values) / len(loss_D_real_values)
    avg_loss_D = sum(loss_D_values) / len(loss_D_values)
    avg_loss_G_GAN = sum(loss_G_GAN_values) / len(loss_G_GAN_values)
    avg_loss_G = sum(loss_G_values) / len(loss_G_values)
    avg_loss_G_LI = sum(loss_G_LI_values) / len(loss_G_LI_values)

    return avg_fid_score, avg_loss_D_fake, avg_loss_D_real, avg_loss_D, avg_loss_G_GAN, avg_loss_G, avg_loss_G_LI

# Evaluate the model
avg_fid, avg_loss_D_fake, avg_loss_D_real, avg_loss_D, avg_loss_G_GAN, avg_loss_G, avg_loss_G_LI = evaluate_model(model, dm)
print(f"Avg FID Score: {avg_fid}")
print(f"Avg Loss D Fake: {avg_loss_D_fake}")
print(f"Avg Loss D Real: {avg_loss_D_real}")
print(f"Avg Loss D: {avg_loss_D}")
print(f"Avg Loss G GAN: {avg_loss_G_GAN}")
print(f"Avg Loss G: {avg_loss_G}")
print(f"Avg Loss G LI: {avg_loss_G_LI}")
