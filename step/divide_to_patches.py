import glob
from os import path

import numpy as np

from tqdm import tqdm

from cityscapes.dataloader import get_divided_image_labels
import logging as l
from functools import partial
from tqdm.contrib.concurrent import process_map
from multiprocessing import Lock

lock = Lock()

def job(image_filename:str, cityscapes_dir:str, dir_str:str, patch_size:int):
    img_np, sem_seg_label_np, hot_label_np, unique_label_np, label_path_before_postfix = get_divided_image_labels(cityscapes_dir, dir_str, patch_size, image_filename)
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            with lock:
                np.save(label_path_before_postfix + f"_{patch_size}_{i}_{j}_img.npy", img_np[i, j])
                np.save(label_path_before_postfix + f"_{patch_size}_{i}_{j}_sem_seg_label.npy", sem_seg_label_np[i, j])
                np.save(label_path_before_postfix + f"_{patch_size}_{i}_{j}_hot_label.npy", hot_label_np[i, j])
                np.save(label_path_before_postfix + f"_{patch_size}_{i}_{j}_unique_label.npy", unique_label_np[i, j])

def divide_to_patches(cityscapes_dir, dir_str, patch_size):
    filename_list = glob.glob(path.join(cityscapes_dir, "leftImg8bit", dir_str, "**", "*_leftImg8bit.png"), recursive=True)
    # for image in tqdm(filename_list):
        # job(image, cityscapes_dir, dir_str, patch_size)
    process_map(partial(job, cityscapes_dir=cityscapes_dir, dir_str=dir_str, patch_size=patch_size), filename_list, max_workers=10, chunksize=32)