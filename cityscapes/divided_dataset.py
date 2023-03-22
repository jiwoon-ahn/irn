from torch.utils.data import Dataset

from enum import Enum, auto

from cityscapes.dataloader import *
from step.divide_to_patches import divide_to_patches
import glob
import typing

class Divide(Enum):
    Train = auto()
    Val = auto()
    Test = auto()

class CityScapesDividedDataset(Dataset):
    _full_width = 2048
    _full_height = 1024
    _cityscapes_dir = "/home/postech2/datasets/cityscapes"

    def __init__(self, divide: Divide, patch_size: int, transform: typing.Union[typing.Callable ,None]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.columns_per_image = self._full_width // patch_size
        self.rows_per_image = self._full_height // patch_size
        self.patches_per_image = self.rows_per_image * self.columns_per_image
        self.transform = transform
        if divide == Divide.Train:
            self.dir_str = "train"
        elif divide == Divide.Test:
            self.dir_str = "test"
        elif divide == Divide.Val:
            self.dir_str = "val"
        self.images_filename = glob.glob(path.join(self._cityscapes_dir, "leftImg8bit", self.dir_str, "**", "*_leftImg8bit.png"), recursive=True)
        
        try:
            self.__getitem__(0)
        except:
            divide_to_patches(self._cityscapes_dir, self.dir_str, self.patch_size)


    def __getitem__(self, index):
        file_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image
        row_index = patch_index // self.columns_per_image
        col_index = patch_index % self.columns_per_image

        image_filename = self.images_filename[file_index]
        basename = path.basename(image_filename)
        image, hot_label, unique_label, sem_seg_label = load_memoized_labels(self._cityscapes_dir, self.dir_str, self.patch_size, row_index, col_index, image_filename)

        if self.transform != None:
            image = self.transform(image)
        
        return image, hot_label, unique_label, basename, row_index, col_index, sem_seg_label

    
    def __len__(self):
        # return 1024
        return len(self.images_filename) * self.rows_per_image * self.columns_per_image

class VOC2012DividedDataset(Dataset):
    _dir = "/home/postech2/irn/VOCdevkit/VOC2012/Divided"

    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform
        self.images_filename = glob.glob(path.join(self._dir, "*_img.npy"))

    def __getitem__(self, index):
        image_filename = self.images_filename[index]
        label_filename = self._get_label_filename(image_filename)

        image = np.load(image_filename)
        hot_label = np.load(label_filename)
        unique_label = hot_to_unique(hot_label, 20)

        if self.transform != None:
            image = self.transform(image)
        
        return image, hot_label, unique_label
    
    def _get_label_filename(self, image_filename):
        _, basename = path.split(image_filename)
        basename, _ = path.splitext(basename)
        return path.join(self._dir, basename.split("_img")[0] + "_label.npy")
    
    def __len__(self):
        return len(self.images_filename)