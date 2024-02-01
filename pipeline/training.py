from cycleGAN_model import CycleGAN
# from augmentation import show_img
from data_loader import CustomDataModule
from generator import UNetGenerator, ResNetGenerator
import os
import pytorch_lightning as L
import torch
import deepspeed as ds


_ = L.seed_everything(0, workers=True)



DEBUG = False

DM_CONFIG = {    
    "monet_dir": os.path.join("/path to dir having monet paintings/", "*.jpg"),
    "photo_dir": os.path.join("/path to dir having pictures/ ", "*.jpg"),
    
    "loader_config": {
        "num_workers": os.cpu_count(),
        "pin_memory": torch.cuda.is_available(),
    },
    "sample_size": 5,
    "batch_size": 1 if not DEBUG else 1,
}
dm_sample = CustomDataModule(batch_size=5, **{k: v for k, v in DM_CONFIG.items() if k != "batch_size"})
dm_sample.setup("fit")
train_loader = dm_sample.train_dataloader()
imgs = next(iter(train_loader))
# show_img(imgs["monet"], nrow=5, title="Augmented Monet Paintings")
# show_img(imgs["photo"], nrow=5, title="Augmented Photos")



MODEL_CONFIG = {
    # the type of generator, and the number of residual blocks if ResNet generator is used
    "gen_name": "unet", # types: 'unet', 'resnet'
    "num_resblocks": 6,
    # the number of filters in the first layer for the generators and discriminators
    "hid_channels": 64,
    # using DeepSpeed's FusedAdam (currently GPU only) is slightly faster
    "optimizer": ds.ops.adam.FusedAdam if torch.cuda.is_available() else torch.optim.Adam,
    # the learning rate and beta parameters for the Adam optimizer
    "lr": 2e-4,
    "betas": (0.5, 0.999),
    # the weights used in the identity loss and cycle loss
    "lambda_idt": 0.5,
    "lambda_cycle": (10, 10), # (MPM direction, PMP direction)
    # the size of the buffer that stores previously generated images
    "buffer_size": 100,
    # the number of epochs for training
    "num_epochs": 18 if not DEBUG else 2,
    # the number of epochs before starting the learning rate decay
    "decay_epochs": 18 if not DEBUG else 1,
}

TRAIN_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    
    # train on 16-bit precision
    "precision": "16-mixed" if torch.cuda.is_available() else 32,
    
    # train on single GPU
    "devices": 1,
    
    # save checkpoint only for last epoch by default
    "enable_checkpointing": True,
    
    # disable logging for simplicity
    "logger": False,
    
    # the number of epochs for training (we limit the number of train/predict batches during debugging)
    "max_epochs": MODEL_CONFIG["num_epochs"],
    "limit_train_batches": 1.0 if not DEBUG else 2,
    "limit_predict_batches": 1.0 if not DEBUG else 5,
    
    # the maximum amount of time for training, in case we exceed run-time of 5 hours
    "max_time": {"hours": 4, "minutes": 55},
    
    # use a small subset of photos for validation/testing (we limit here for flexibility)
    "limit_val_batches": 1,
    "limit_test_batches": 5,
    
    # disable sanity check before starting the training routine
    "num_sanity_val_steps": 0,
    
    # the frequency to visualize the progress of adding Monet style
    "check_val_every_n_epoch": 6 if not DEBUG else 1,
}

dm = CustomDataModule(**DM_CONFIG)
model = CycleGAN(**MODEL_CONFIG)
trainer = L.Trainer(**TRAIN_CONFIG)
trainer.fit(model, datamodule=dm)

torch.save(model.state_dict(), "cycle_gan_model.pth")