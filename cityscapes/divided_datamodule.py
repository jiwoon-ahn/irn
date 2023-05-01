import pytorch_lightning as pl
import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from torch.utils.data import DataLoader
from torchvision import transforms

from step.divide_to_patches import divide_to_patches
from cityscapes.divided_dataset import CityScapesDividedDataset, Divide

from cityscapesscripts.preparation import createTrainIdLabelImgs
from cityscapes.dataset import MultipleScalesTranform

from cityscapes.divided_dataset import CityScapesDividedPTHCAMDataset

class CityScapesDividedModule(pl.LightningDataModule):
    _cityscapes_dir = "/workspaces/datasets/cityscapes"
    def __init__(self, batch_size: int, patch_size: int, crop_size: int, prepare_trainid: bool, prepare_patches: bool, cam_crop_size: int, cam_scales: list[float], cam_out_dir:str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.prepare_trainid = prepare_trainid
        self.prepare_patches = prepare_patches
        self.cam_crop_size = cam_crop_size
        self.cam_scales = cam_scales
        self.cam_out_dir = cam_out_dir

    def prepare_data(self) -> None:
        if self.prepare_trainid:
            os.environ['CITYSCAPES_DATASET'] = self._cityscapes_dir
            createTrainIdLabelImgs.main()

        if self.prepare_patches:
            divide_to_patches(self._cityscapes_dir, "train", self.patch_size)
            divide_to_patches(self._cityscapes_dir, "test", self.patch_size)
            divide_to_patches(self._cityscapes_dir, "val", self.patch_size)
    
    def setup(self, stage: str) -> None:
        if self.prepare_patches:
            if stage == 'fit':
                divide_to_patches(self._cityscapes_dir, "train", self.patch_size)
            if stage == 'test':
                divide_to_patches(self._cityscapes_dir, "test", self.patch_size)
            if stage == 'validate':
                divide_to_patches(self._cityscapes_dir, "val", self.patch_size)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(self.crop_size, antialias=False), # 넣었다 뺐다
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = CityScapesDividedDataset(Divide.Train, ["img", "hot_label"], self.patch_size, transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.crop_size + 10, antialias=False), # type: ignore
            transforms.CenterCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = CityScapesDividedDataset(
            Divide.Val, 
            ["img", "hot_label", "unique_label"],
            self.patch_size, 
            transform=transform
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((self.cam_crop_size, self.cam_crop_size),antialias=False), # type: ignore
            MultipleScalesTranform(self.cam_scales)]
        )
        dataset =  CityScapesDividedDataset(
            divide=Divide.Val,
            datatype=["img"],
            patch_size=self.patch_size,
            transform=transform
        )
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((self.cam_crop_size, self.cam_crop_size), transforms.InterpolationMode.NEAREST),
        ])
        dataset = CityScapesDividedPTHCAMDataset(
            divide=Divide.Val,
            patch_size=self.patch_size,
            cam_out_dir=self.cam_out_dir,
            transform=transform
        )
        return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)