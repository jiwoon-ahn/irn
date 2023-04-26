import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms

from step.divide_to_patches import divide_to_patches
from cityscapes.divided_dataset import CityScapesDividedDataset, Divide

class CityScapesDividedModule(pl.LightningDataModule):
    _cityscapes_dir = "/workspaces/datasets/cityscapes"
    def __init__(self, batch_size: int, patch_size: int, crop_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.crop_size = crop_size

    # def prepare_data(self) -> None:
        # divide_to_patches(self._cityscapes_dir, "train", self.patch_size)
        # divide_to_patches(self._cityscapes_dir, "test", self.patch_size)
        # divide_to_patches(self._cityscapes_dir, "val", self.patch_size)

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
