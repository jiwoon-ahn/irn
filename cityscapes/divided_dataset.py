from torch.utils.data import Dataset, DataLoader

from enum import Enum, auto

from cityscapes.dataloader import *
from step.divide_to_patches import divide_to_patches
import glob
import typing
import torch
from torch import nn
from torchvision.transforms import Resize, InterpolationMode
import cv2
import pytorch_lightning as pl

class Divide(Enum):
    Train = auto()
    Val = auto()
    Test = auto()

class CityScapesDividedDataset(Dataset):
    _full_width = 2048
    _full_height = 1024
    _cityscapes_dir = "/Users/minsu/dataset/cityscapes"
    _cityscapes_preprocessed_dir = "/Users/minsu//dataset/citiscape_preprocessed"

    def __init__(self, divide: Divide, datatype: list[str], patch_size: int, transform: typing.Optional[typing.Callable]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.columns_per_image = self._full_width // patch_size
        self.rows_per_image = self._full_height // patch_size
        self.patches_per_image = self.rows_per_image * self.columns_per_image
        self.transform = transform
        self.datatype = datatype
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
        result = load_memoized_labels(self._cityscapes_dir, self.dir_str, self.patch_size, row_index, col_index, image_filename, self.datatype)

        if self.transform != None and "img" in self.datatype:
            index_to_transform = self.datatype.index("img")
            result[index_to_transform] = self.transform(result[index_to_transform])
        
        return basename, row_index, col_index, *result

    
    def __len__(self):
        # return 1024
        return len(self.images_filename) * self.rows_per_image * self.columns_per_image
    
class CityScapesDividedCAMDataset(Dataset):
    _full_width = 2048
    _full_height = 1024
    _cityscapes_dir = "/Users/minsu/dataset/cityscapes"

    def __init__(self, divide: Divide, patch_size: int, cam_out_dir: str, cam_eval_thres:float, cam_size:int, transform: typing.Union[typing.Callable ,None]) -> None:
        self._cam_out_dir = cam_out_dir
        self._dataset = CityScapesDividedDataset(divide,["sem_seg_label"], patch_size, transform)
        self._cam_eval_thres = cam_eval_thres
        self._cam_size = cam_size
        self._resize = Resize((cam_size, cam_size), interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        (basename, row_index, col_index, sem_seg_label) = self._dataset[index]
        basename = path.splitext(basename)[0]
        filename_prefix = path.join(self._cam_out_dir, f'{basename}_{row_index}_{col_index}')

        cams = np.moveaxis(np.load(filename_prefix + '_highres.npy'), 0, -1)
        keys = np.load(filename_prefix + '_valid_cat.npy')

        cams_sparse = np.zeros((cams.shape[0], cams.shape[1], 20), dtype=np.float32)
        cams_sparse[:, :, keys] = cams

        return cams_sparse, sem_seg_label

    def __len__(self):
        return len(self._dataset)
    

class CityScapesDividedPTHCAMDataset(Dataset):
    _full_width = 2048
    _full_height = 1024
    _cityscapes_dir = "/Users/minsu/dataset/cityscapes"

    def __init__(self, divide: Divide, patch_size: int, cam_out_dir: str, cam_size:int, transform: typing.Union[typing.Callable ,None]) -> None:
        self._cam_out_dir = cam_out_dir
        self._dataset = CityScapesDividedDataset(divide, ["sem_seg_label"], patch_size, transform)
        self._cam_size = cam_size
        self._resize = Resize((cam_size, cam_size), interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, index):
        (basename, row_index, col_index, sem_seg_label) = self._dataset[index]
        basename = path.splitext(basename)[0]
        filename_prefix = path.join(self._cam_out_dir, f'{basename}_{row_index}_{col_index}')

        cams = np.load(filename_prefix + '_highres.npy')

        return cams, sem_seg_label

    def __len__(self):
        return len(self._dataset)

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