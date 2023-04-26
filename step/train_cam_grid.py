import torch
from torch.backends import cudnn
cudnn.enabled = True

from torchvision import transforms

import cityscapes.divided_dataset
from cityscapes.divided_datamodule import CityScapesDividedModule
from net.resnet50_cam_lightning import Net

import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy


def run(args):
    num_workers = args.num_workers
    crop_size = args.cam_crop_size
    batch_size = args.cam_batch_size
    num_epoches = args.cam_num_epoches
    learning_rate = args.cam_learning_rate
    weight_decay = args.cam_weight_decay
    patch_size = args.patch_size
    cam_weights_name = args.cam_weights_name
    
    model = Net(learning_rate, weight_decay, 0.5)
    datamodule = CityScapesDividedModule(batch_size, patch_size, crop_size)

    logger = WandbLogger(project="irn-cityscapes", name="train_cam_grid_no_sigmoid")
    logger.log_hyperparams(args)
    checkpoint_callback = ModelCheckpoint(dirpath='models/train', filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}' )
    
    trainer = pl.Trainer(logger=logger, max_epochs=num_epoches, callbacks=[checkpoint_callback],strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model, datamodule , )
    
    savefile_name = cam_weights_name + '.pth'
    save_model(model, savefile_name)
    torch.cuda.empty_cache()
    wandb.finish()

def save_model(model, savefile_name):
    torch.save(model.module.state_dict(), savefile_name)
    wandb.save(savefile_name)

def get_cityscapes_datamodule(patch_size, batch_size, crop_size, num_workers):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(crop_size, antialias=False), # 넣었다 뺐다
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(crop_size + 10, antialias=False), # type: ignore
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = cityscapes.divided_dataset.CityScapesDividedDataset(
        cityscapes.divided_dataset.Divide.Train,
        ["img", "hot_label"],
        int(patch_size),
        transform=train_transform
    )
    # val_dataset = cityscapes.divided_dataset.CityScapesDividedDataset(
    #     cityscapes.divided_dataset.Divide.Val, 
    #     ["img", "hot_label", "unique_label"],
    #     int(patch_size), 
    #     transform=val_transform
    # )

    return pl.LightningDataModule.from_datasets(train_dataset, batch_size=batch_size, num_workers=num_workers)
