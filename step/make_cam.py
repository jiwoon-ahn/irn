import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
cudnn.enabled = True

import numpy as np
import importlib
import os
import wandb

from cityscapes.dataset import Cityscapes, MultipleScalesTranform
from cityscapes.divided_dataset import CityScapesDividedDataset, Divide
import cityscapesscripts.helpers.labels as labels
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage, Lambda, PILToTensor
from misc import torchutils, imutils


from tqdm import tqdm

from functools import partial
from tqdm.contrib.concurrent import process_map

import os.path as path


def _work(model, dataset, args):

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():

        for basename, row_index, col_index, images in tqdm(data_loader):
            size = (args.cam_crop_size, args.cam_crop_size)
            # strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            num_inputs = images[0].shape[0]

            # strided_cam = torch.zeros((num_inputs, 20) + strided_size, device='cuda')
            highres_cam = torch.zeros((num_inputs, 20) + strided_up_size, device='cuda')
            for img in images:
                output = model(img.cuda())
                # strided_cam += F.interpolate(output, strided_size, mode='bilinear', align_corners=False)
                highres_cam += F.interpolate(output, strided_up_size, mode='bilinear', align_corners=False)
            
            # strided_cam /= (F.adaptive_max_pool2d(strided_cam.flatten(start_dim=0, end_dim=1), (1, 1)) + 1e-5).unflatten(0, (-1, 20))
            highres_cam /= (F.adaptive_max_pool2d(highres_cam.flatten(start_dim=0, end_dim=1), (1, 1)) + 1e-5).unflatten(0, (-1, 20))

            for j in range(num_inputs):
                # strided_cam__ = strided_cam[j]
                highres_cam__ = highres_cam[j].detach().cpu().numpy()
                basename__ = path.splitext(basename[j])[0]

                filename_prefix = os.path.join(args.cam_out_dir, f'{basename__}_{row_index[j].item()}_{col_index[j].item()}')
                # torch.save(strided_cam__, filename_prefix + '_strided.pth')
                np.save(filename_prefix + '_highres.npy', highres_cam__)

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load("/home/postech2/irn/wandb/run-20230223_211046-jfnt9scc/files/sess/res50_cam.pth.pth"), strict=True)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    dataset =  CityScapesDividedDataset(
        divide=Divide.Val,
        datatype=["img"],
        patch_size=args.patch_size,
        transform=Compose([ToTensor(), Resize((args.cam_crop_size, args.cam_crop_size)), MultipleScalesTranform(args.cam_scales)])
    )

    _work(model, dataset, args)

    torch.cuda.empty_cache()