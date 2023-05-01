from torch.backends import cudnn
cudnn.enabled = True

from cityscapes.divided_datamodule import CityScapesDividedModule
from net.resnet50_cam_lightning import CAM

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

def run(args):
    crop_size = args.cam_crop_size
    batch_size = args.cam_batch_size
    learning_rate = args.cam_learning_rate
    weight_decay = args.cam_weight_decay
    patch_size = args.patch_size
    
    model = CAM(learning_rate, weight_decay, 0.5, args.cam_out_dir, crop_size)
    datamodule = CityScapesDividedModule(batch_size, patch_size, crop_size, False, False, args.cam_crop_size, args.cam_scales, args.cam_out_dir)

    logger = WandbLogger(project="irn-cityscapes", name="train_cam_grid_no_sigmoid")
    logger.log_hyperparams(args)
    
    trainer = pl.Trainer(devices=1, num_nodes=1, logger=logger)
    trainer.test(model, datamodule)