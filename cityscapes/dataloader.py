import glob
import numpy as np

from torch.utils.data import Dataset
from os import path

from enum import Enum, auto
from functools import lru_cache
import typing

from tqdm import tqdm

import cv2

class Divide(Enum):
    Train = auto()
    Val = auto()
    Test = auto()



class CityScapesDividedDataset(Dataset):
    _full_width = 2048
    _full_height = 1024
    _cityscapes_dir = "/home/postech2/datasets/cityscapes"

    def __init__(self, divide: Divide, patch_size: int, transform: typing.Callable) -> None:
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

    def __getitem__(self, index):
        file_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image
        row_index = patch_index // self.columns_per_image
        col_index = patch_index % self.columns_per_image

        image_filename = self.images_filename[file_index]
        imgs_np, hot_labels_np, unique_labels_np = self._get_divided_image_labels(image_filename)
        image = imgs_np[row_index, col_index]
        hot_label = hot_labels_np[row_index, col_index]
        unique_label = unique_labels_np[row_index, col_index]

        if self.transform != None:
            image = self.transform(image)
        
        return image, hot_label, unique_label
    
    @lru_cache(maxsize=32)
    def _get_divided_image_labels(self, image_filename: str):
        train_id_filename = self._get_train_id_filename(image_filename)

        loded_img = cv2.imread(image_filename)
        loaded_label = cv2.imread(train_id_filename)

        img_np = array_to_divided_array(loded_img, self.patch_size)
        hot_label_np = array_to_divided_multi_hot_2d_trainid(loaded_label, self.patch_size, 20)
        unique_label_np = array_to_divided_unique_2d(loaded_label, self.patch_size, 20)
        return img_np, hot_label_np, unique_label_np
    
    def _get_train_id_filename(self, image_filename):
        dirname, basename = path.split(image_filename)
        basename, _ = path.splitext(basename)
        base_dirname = path.basename(dirname)
        train_id_base_filename = basename.split("_leftImg8bit")[0] + "_gtFine_labelTrainIds.png"
        return path.join(self._cityscapes_dir, "gtFine", self.dir_str, base_dirname, train_id_base_filename)
    
    def __len__(self):
        # return 1024
        return len(self.images_filename) * self.rows_per_image * self.columns_per_image
    
    @lru_cache()
    def class_distribution(self):
        result = np.zeros(shape=(20,))
        for f in tqdm(self.images_filename, desc="loading class distribution"):
            _, hot_label_np, _ = self._get_divided_image_labels(f)
            result += np.count_nonzero(hot_label_np, axis=(0, 1))
        return result


class VOC2012DividedDataset(Dataset):
    _dir = "/home/postech2/irn/VOCdevkit/VOC2012/Divided"

    def __init__(self, transform: typing.Callable) -> None:
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
    

# Converts 2d array to 3d array
# Array is divided into (m * n) rectangular sections,
# and multi-hot encoded by checking existance of elements on original array.
# arr.shape: (m * divide_size, n * divide_size)
# return.shape: (m, n, multi_hot_slot_size)
def array_to_divided_multi_hot_2d_trainid(
    arr: np.ndarray,
    divide_size: int,
    multi_hot_slot_size: int, 
) -> np.ndarray:
    assert (arr[:, :, 0]==arr[:, :, 1]).all()
    assert (arr[:, :, 1]==arr[:, :, 2]).all()
    arr = arr[:, :, 0].copy()
    arr[arr == -1] = multi_hot_slot_size - 1
    arr[arr == 255] = multi_hot_slot_size - 1
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, multi_hot_slot_size), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            divided_image = arr[r:r+divide_size, c:c+divide_size]
            result[i, j, divided_image] = 1
    return result
    

# arr.shape: (m * divide_size, n * divide_size)
# return.shape: (m, n, divide_size, divide_size)
def array_to_divided_array(
    arr: np.ndarray,
    divide_size: int,
) -> np.ndarray:
    m, n, l = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, divide_size, divide_size, l), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            result[i, j, :, :, :] = arr[r:r+divide_size, c:c+divide_size]
    return result

def array_to_divided_unique_2d(
    arr: np.ndarray,
    divide_size: int,
    num_classes: int, 
) -> np.ndarray:
    assert (arr[:, :, 0]==arr[:, :, 1]).all()
    assert (arr[:, :, 1]==arr[:, :, 2]).all()
    arr = arr[:, :, 0].copy()
    arr[arr == -1] = num_classes
    arr[arr == 255] = num_classes
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, num_classes), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            divided_image = arr[r:r+divide_size, c:c+divide_size]
            uniques = np.unique(divided_image)
            if uniques.shape[0] < num_classes:
                uniques = np.hstack((uniques, np.full((num_classes - uniques.shape[0]), -1, dtype=np.int32)))
            result[i, j] = uniques
    return result

def hot_to_unique(
    arr: np.ndarray,
    num_classes: int, 
) -> np.ndarray:
    result = np.full((num_classes, ), -1, dtype=np.int64)
    (uniques, ) = np.nonzero(arr)
    if uniques.shape[0] > 0:
        result[:uniques.shape[0]]
    return result

# Converts 2d array to 3d array
# Array is divided into (m * n) rectangular sections,
# and class distributions normalized to 0-1 by checking existance of elements on original array.
# arr.shape: (m * divide_size, n * divide_size)
# return.shape: (m, n, multi_hot_slot_size)
def array_to_divided_class_distributions_trainid(
    arr: np.ndarray,
    divide_size: int,
    multi_hot_slot_size: int, 
) -> np.ndarray:
    assert (arr[:, :, 0]==arr[:, :, 1]).all()
    assert (arr[:, :, 1]==arr[:, :, 2]).all()
    arr = arr[:, :, 0].copy()
    arr[arr == -1] = multi_hot_slot_size - 1
    arr[arr == 255] = multi_hot_slot_size - 1
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, multi_hot_slot_size), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            divided_image = arr[r:r+divide_size, c:c+divide_size]
            result[i, j, divided_image] = 1
    return result
    