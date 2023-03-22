
import numpy as np
import os
import cityscapes.divided_dataset as cityscapes
from torchvision.transforms import Compose, ToTensor, Resize
from ignite.metrics import ConfusionMatrix, IoU, mIoU, Precision
from torchvision.transforms.functional import resize, InterpolationMode
import torch
from torch import nn
from tqdm import tqdm
from os import path

def run(args):
    dataset = cityscapes.CityScapesDividedDataset(divide=cityscapes.Divide.Test, patch_size=args.patch_size, transform=Compose([ToTensor(), Resize((args.cam_crop_size, args.cam_crop_size)),]))

    confusion = ConfusionMatrix(20)
    precision = Precision(is_multilabel=True)
    micro_precision = Precision(is_multilabel=True, average="micro")
    macro_precision = Precision(is_multilabel=True, average="macro")

    for (_, _, _, basename, row_index, col_index, sem_seg_label) in (pbar := tqdm(dataset)):
        sem_seg_label = resize(torch.tensor(sem_seg_label, dtype=torch.long).unsqueeze(0), [args.cam_crop_size, args.cam_crop_size], InterpolationMode.NEAREST).squeeze()
        basename = path.splitext(basename)[0]
        filename_prefix = os.path.join(args.cam_out_dir, f'{basename}_{row_index}_{col_index}')

        # np.load(filename_prefix + '_strided.npy')
        cams = torch.tensor(np.load(filename_prefix + '_highres.npy').squeeze(0)).moveaxis(0, -1)
        keys = torch.tensor(np.load(filename_prefix + '_valid_cat.npy'))
        # cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        cls_labels = torch.zeros((cams.shape[0], cams.shape[1], 20), dtype=torch.int8)
        cls_labels[:, :, keys] = (cams > args.cam_eval_thres).to(dtype=torch.int8)
        # cls_labels = np.argmax(cams, axis=0)
        # cls_labels = keys[cls_labels]
        # cls_labels[cls_labels > 19] = 19
        sem_seg_label = nn.functional.one_hot(sem_seg_label, 20)
        precision.update((cls_labels.flatten(start_dim=0, end_dim=1), sem_seg_label.flatten(start_dim=0, end_dim=1)))
        micro_precision.update((cls_labels.flatten(start_dim=0, end_dim=1), sem_seg_label.flatten(start_dim=0, end_dim=1)))
        macro_precision.update((cls_labels.flatten(start_dim=0, end_dim=1), sem_seg_label.flatten(start_dim=0, end_dim=1)))
    
    print({'micro_precision': micro_precision.compute(), 'macro_precision': macro_precision.compute()})
