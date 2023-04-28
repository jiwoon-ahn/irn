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
    wandb.init()
    crop_size = wandb.config.cam_crop_size
    batch_size = wandb.config.cam_batch_size
    num_epoches = wandb.config.cam_num_epoches
    learning_rate = wandb.config.cam_learning_rate
    weight_decay = wandb.config.cam_weight_decay
    patch_size = wandb.config.patch_size
    
    model = CAM(learning_rate, weight_decay, 0.5, args.cam_out_dir, crop_size)
    datamodule = CityScapesDividedModule(batch_size, patch_size, crop_size, False, False, args.cam_crop_size, args.cam_scales, args.cam_out_dir)

    logger = WandbLogger()
    logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(save_top_k=10, monitor='val_macro_precision', mode='max', dirpath='models/train', filename='{epoch}-{val_loss:.2f}-{val_macro_precision:.2f}-{val_micro_precision:.2f}' )
    
    trainer = pl.Trainer(logger=logger, max_epochs=num_epoches, callbacks=[checkpoint_callback, FeatureExtractorFreezeUnfreeze()],strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model, datamodule)