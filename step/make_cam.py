import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import wandb

from cityscapes.dataset import Cityscapes, MultipleScalesTranform
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage, Lambda, PILToTensor
from misc import torchutils, imutils

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    user = "postech-cv"
    project = "irn-cityscapes"
    display_name = "make_cam"

    wandb.init(
        entity=user,
        project=project,
        name=display_name,
        config={
            "cam": {
                "network": args.cam_network,
                "crop_size": args.cam_crop_size,
                "batch_size": args.cam_batch_size,
                "num_epoches": args.cam_num_epoches,
                "learning_rate": args.cam_learning_rate,
                "weight_decay": args.cam_weight_decay,
                "eval_thres": args.cam_eval_thres,
                "scales": args.cam_scales,
                "patch_size": args.patch_size,
            }
        }
    )

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        model.eval()

        for iter, (image, label, img_name) in enumerate(data_loader):
            size = (256, 512)
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in image]

            strided_cam = torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False).squeeze(0) for o
                 in outputs], 0)
            strided_cam = torch.sum(strided_cam, 0)
            wandb.log({
                "strided cam": wandb.Image(ToPILImage()(strided_cam))
            })

            highres_cam = torch.stack(
                [F.interpolate(torch.unsqueeze(o, 1), strided_up_size, mode='bilinear', align_corners=False).squeeze(0) for o
                 in outputs], 0)
            highres_cam = torch.sum(highres_cam, 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = Cityscapes(
        "/home/postech2/datasets/cityscapes", 
        split='val', 
        mode='fine',
        target_type='semantic', 
        transform=Compose([ToTensor(), Resize((128, 256)), MultipleScalesTranform(args.cam_scales)]), 
        target_transform=Compose([PILToTensor(), Resize(256)])
    )
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    _work(0, model, dataset, args)
    # multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()