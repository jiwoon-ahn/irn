import torch
from torch.backends import cudnn
from torch import distributed
cudnn.enabled = True

from cityscapes.divided_datamodule import CityScapesDividedModule
from net.resnet50_cam_lightning import CAM, FeatureExtractorFreezeUnfreeze

import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

def run(args):
    wandb.init(project="irn-cityscapes", name="train_cam_grid_no_sigmoid", config=args)
    crop_size = wandb.config.cam_crop_size
    batch_size = wandb.config.cam_batch_size
    learning_rate = wandb.config.cam_learning_rate
    weight_decay = wandb.config.cam_weight_decay
    patch_size = wandb.config.patch_size
    
    model = CAM(learning_rate, weight_decay, 0.5, args.cam_out_dir, crop_size)
    datamodule = CityScapesDividedModule(batch_size, patch_size, crop_size, False, False, args.cam_crop_size, args.cam_scales, args.cam_out_dir)

    logger = WandbLogger()
    logger.log_hyperparams(args)
    
    trainer = pl.Trainer(logger=logger,strategy=DDPStrategy(find_unused_parameters=True))
    trainer.predict(model, datamodule, ckpt_path="/workspaces/irn/models/train/epoch=4-val_loss=-0.39-val_macro_precision=0.67-val_micro_precision=0.85.ckpt")