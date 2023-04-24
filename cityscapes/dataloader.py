import numpy as np
from os import path
from functools import lru_cache
import cv2

def load_memoized_labels(cityscapes_dir, dir_str, patch_size, row_index, col_index, image_filename, datatype:list[str]):
    label_path_before_postfix = get_label_path_before_postfix(cityscapes_dir, dir_str, image_filename)
    prefix = f"{label_path_before_postfix}_{patch_size}_{row_index}_{col_index}"
    result = []
    for d in datatype:
        d_path = prefix + f"_{d}.npy"
        result.append(np.load(d_path, allow_pickle=True))
    return result
    img_path = prefix + "_img.npy"
    hot_label_path = prefix + "_hot_label.npy"
    unique_label_path = prefix + "_unique_label.npy"
    sem_seg_label_path = prefix + "_sem_seg_label.npy"

    image = np.load(img_path, allow_pickle=True)
    hot_label = np.load(hot_label_path, allow_pickle=True)
    unique_label = np.load(unique_label_path, allow_pickle=True)
    sem_seg_label = np.load(sem_seg_label_path, allow_pickle=True)
    return image,hot_label,unique_label,sem_seg_label

@lru_cache(maxsize=32)
def get_divided_image_labels(cityscapes_dir:str, dir_str:str, patch_size:int, image_filename: str):
    label_path_before_postfix = get_label_path_before_postfix(cityscapes_dir, dir_str, image_filename)
    train_id_filename = label_path_before_postfix + "_gtFine_labelTrainIds.png"

    loded_img = cv2.imread(image_filename)
    loaded_label = cv2.imread(train_id_filename, cv2.IMREAD_GRAYSCALE)

    img_np = array_to_divided_array(loded_img, patch_size)
    sem_seg_label_np = array_to_divided_array_single_channel(loaded_label, patch_size)
    sem_seg_label_np[sem_seg_label_np == -1] = 19
    sem_seg_label_np[sem_seg_label_np == 255] = 19
    hot_label_np = array_to_divided_multi_hot_2d_trainid(loaded_label, patch_size, 20)
    unique_label_np = array_to_divided_unique_2d(loaded_label, patch_size, 20)
    return img_np, sem_seg_label_np, hot_label_np, unique_label_np, label_path_before_postfix

def get_label_path_before_postfix(cityscapes_dir, dir_str, image_filename):
    dirname, basename = path.split(image_filename)
    basename, _ = path.splitext(basename)
    base_dirname = path.basename(dirname)
    train_id_base_filename = basename.split("_leftImg8bit")[0]
    return path.join(cityscapes_dir, "gtFine", dir_str, base_dirname, train_id_base_filename)

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
    arr[arr == -1] = multi_hot_slot_size - 1
    arr[arr == 255] = multi_hot_slot_size - 1
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, multi_hot_slot_size), dtype=np.byte)
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
    result = np.zeros((m,n, divide_size, divide_size, l), dtype=np.ubyte)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            result[i, j, :, :, :] = arr[r:r+divide_size, c:c+divide_size]
    return result

# arr.shape: (m * divide_size, n * divide_size)
# return.shape: (m, n, divide_size, divide_size)
def array_to_divided_array_single_channel(
    arr: np.ndarray,
    divide_size: int,
) -> np.ndarray:
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m, n, divide_size, divide_size), dtype=arr.dtype)
    for i in range(m):
        for j in range(n):
            r = i * divide_size
            c = j * divide_size
            result[i, j, :, :] = arr[r:r+divide_size, c:c+divide_size]
    return result

def array_to_divided_unique_2d(
    arr: np.ndarray,
    divide_size: int,
    num_classes: int, 
) -> np.ndarray:
    arr[arr == -1] = num_classes
    arr[arr == 255] = num_classes
    m, n = arr.shape
    m = m // divide_size
    n = n // divide_size
    result = np.zeros((m,n, num_classes), dtype=np.int32)
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