import cityscapes.divided_dataset as cityscapes
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import resize, InterpolationMode
from torch import nn
import torch
import typing
import pytorch_lightning as pl
from pytorch_lightning.utilities import types
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.classification import MultilabelPrecision

class Module(pl.LightningModule):
    def __init__(self, threshold: float, cam_crop_size: int = 256):
        super().__init__()
        self.cam_crop_size = cam_crop_size
        self.micro_precision = MultilabelPrecision(num_labels=20, average='micro', threshold=threshold)
        self.macro_precision = MultilabelPrecision(num_labels=20, average='macro', threshold=threshold)

    def test_step(self, batch, _) -> typing.Optional[types.STEP_OUTPUT]:
        cams_sparse, sem_seg_label = batch

        cams_sparse = cams_sparse.moveaxis(1, -1).flatten(0, 2)

        sem_seg_label = sem_seg_label.to(dtype=torch.int64)
        sem_seg_label = resize(sem_seg_label, [self.cam_crop_size, self.cam_crop_size], InterpolationMode.NEAREST)
        sem_seg_label = sem_seg_label.flatten()
        sem_seg_label = nn.functional.one_hot(sem_seg_label, num_classes=20)
        
        self.micro_precision(cams_sparse, sem_seg_label)
        self.macro_precision(cams_sparse, sem_seg_label)
        self.log_dict({'running_micro_precision': self.micro_precision, 'running_macro_precision': self.macro_precision})

    def on_test_epoch_end(self):
        self.log_dict({'micro_precision': self.micro_precision.compute(), 'macro_precision': self.macro_precision.compute()})

    def test_dataloader(self) -> types.EVAL_DATALOADERS:
        return super().test_dataloader()

def run(args):
    logger = WandbLogger(project="cam", name="cam_eval")
    dataset = cityscapes.CityScapesDividedPTHCAMDataset(
        divide=cityscapes.Divide.Val,
        patch_size=args.patch_size,
        cam_out_dir=args.cam_out_dir,
        cam_size=args.cam_crop_size,
        transform=Compose([ToTensor(), Resize((args.cam_crop_size, args.cam_crop_size), InterpolationMode.NEAREST),])
    )
    lightning_data_module = pl.LightningDataModule.from_datasets(train_dataset=dataset, val_dataset=dataset, test_dataset=dataset, batch_size=128, num_workers=args.num_workers)
    
    model = Module(args.cam_eval_thres, args.cam_crop_size)

    trainer = pl.Trainer(devices=1, num_nodes=1, limit_test_batches=3, logger=logger)
    trainer.test(model=model, datamodule=lightning_data_module)
