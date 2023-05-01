from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch
from torch import optim
import numpy as np
import os
import os.path as path

import pytorch_lightning as pl
from pytorch_lightning.utilities import types
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

import typing

from torchmetrics.classification import MultilabelPrecision
import cityscapesscripts.helpers.labels as labels
import wandb

from torchvision.transforms.functional import resize, InterpolationMode

from misc import torchutils, imutils

class CAM(pl.LightningModule):
    num_classes = 20
    num_examples = 8

    def __build_model__(self):
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1
        )
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

        self.training_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.validation_loss = torch.nn.MultiLabelSoftMarginLoss()

    def __init__(self, learning_rate: float, weight_decay: float, threshold: float, cam_out_dir: str, cam_crop_size: int):
        super(CAM, self).__init__()

        self.__build_model__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.cam_out_dir = cam_out_dir
        self.cam_crop_size = cam_crop_size

        self.training_loss = torch.nn.MultiLabelSoftMarginLoss()
        self.validation_loss = torch.nn.MultiLabelSoftMarginLoss()

        self.val_precision = MultilabelPrecision(self.num_classes, threshold=self.threshold)
        self.val_micro_precision = MultilabelPrecision(self.num_classes, threshold=self.threshold, average='micro')
        self.val_macro_precision = MultilabelPrecision(self.num_classes, threshold=self.threshold, average='macro')
        

        self.predict_micro_precision = MultilabelPrecision(num_labels=20, average='micro', threshold=threshold)
        self.predict_macro_precision = MultilabelPrecision(num_labels=20, average='macro', threshold=threshold)

        self.mode = "classifier"


    def forward(self, x):
        if self.mode == "classifier":
            return self.__forward_classifier(x)
        elif self.mode == "cam":
            return self.__forward_cam(x)
    
    def __forward_classifier(self, x):
        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)
        return x
    
    def __forward_cam(self, x):
        out0 = self.__forward_cam_internal(x)
        out1 = self.__forward_cam_internal(x.flip(-1))
        return out0 + out1.flip(-1)
    
    def __forward_cam_internal(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.classifier(x)
        x = F.relu(x)

        return x

    def newly_added_params(self) -> typing.List[torch.nn.parameter.Parameter]:
        return [p for p in self.newly_added.parameters() if p.requires_grad]

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.SGD(self.newly_added_params(), self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2, step_size_down=3, mode='triangular')
        return [optimizer], [scheduler]
    
    def on_train_start(self) -> None:
        self.mode = "classifier"
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def training_step(self, batch, _) -> types.STEP_OUTPUT:
        _, _, _, img, hot_label = batch
        pred = self(img)
        loss = self.training_loss(pred[:, :(self.num_classes - 1)], hot_label[:, :(self.num_classes - 1)])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_start(self) -> None:
        self.mode = "classifier"

    def validation_step(self, batch, _) -> typing.Optional[types.STEP_OUTPUT]:
        _, _, _, img, hot_label, unique_label = batch
        pred = self(img)
        loss = self.validation_loss(pred, unique_label)

        self.val_precision(pred, hot_label)
        self.val_macro_precision(pred, hot_label)
        self.val_micro_precision(pred, hot_label)

        thresholded_pred = (pred > 0.5).to(dtype=torch.long)

        example_images: typing.List[wandb.Image] = []
        if len(example_images) < self.num_examples:
            for i in range(min(self.num_examples - len(example_images), img.shape[0])):
                pred_str = hot_to_cityscape_str(thresholded_pred[i], self.num_classes)
                label_str = hot_to_cityscape_str(hot_label[i], self.num_classes)
                im = wandb.Image(img[i], caption=f"[prediction]: {pred_str}\n[label]: {label_str}")
                example_images.append(im)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_precision", self.val_precision, sync_dist=True)
        self.log("val_macro_precision", self.val_macro_precision, sync_dist=True)
        self.log("val_micro_precision", self.val_micro_precision, sync_dist=True)
        self.logger.experiment.log({"val_example_images": example_images}) # type: ignore
        return {"val_loss": loss, "val_macro_precision": self.val_macro_precision, "val_micro_precision": self.val_micro_precision}

    def on_predict_start(self) -> None:
        self.mode = "cam"

    def predict_step(self, batch, _) -> STEP_OUTPUT | None:
        _, _, _, images = batch
        size = (self.cam_crop_size, self.cam_crop_size)
        strided_up_size = imutils.get_strided_up_size(size, 16)
        num_inputs = images[0].shape[0]

        highres_cam = torch.zeros((num_inputs, 20) + strided_up_size, device=self.device)
        for img in images:
            output = self(img)
            highres_cam += F.interpolate(output, strided_up_size, mode='bilinear', align_corners=False)
        
        highres_cam /= (F.adaptive_max_pool2d(highres_cam.flatten(start_dim=0, end_dim=1), (1, 1)) + 1e-5).unflatten(0, (-1, 20))
        return highres_cam

    def on_predict_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        basename, row_index, col_index, images = batch
        num_inputs = images[0].shape[0]
        for j in range(num_inputs):
            highres_cam__ = outputs[j].detach().cpu().numpy()
            basename__ = path.splitext(basename[j])[0]

            filename_prefix = os.path.join(self.cam_out_dir, f'{basename__}_{row_index[j].item()}_{col_index[j].item()}')
            np.save(filename_prefix + '_highres.npy', highres_cam__)

    def on_test_start(self) -> None:
        self.mode = "cam"
    
    def test_step(self, batch, _) -> typing.Optional[types.STEP_OUTPUT]:
        cams_sparse, sem_seg_label = batch

        cams_sparse = cams_sparse.moveaxis(1, -1).flatten(0, 2)

        sem_seg_label = sem_seg_label.to(dtype=torch.int64)
        sem_seg_label = resize(sem_seg_label, [self.cam_crop_size, self.cam_crop_size], InterpolationMode.NEAREST)
        sem_seg_label = sem_seg_label.flatten()
        sem_seg_label = nn.functional.one_hot(sem_seg_label, num_classes=20)
        
        self.predict_micro_precision(cams_sparse, sem_seg_label)
        self.predict_macro_precision(cams_sparse, sem_seg_label)
        self.log("predict_micro_precision", self.predict_micro_precision, sync_dist=True)
        self.log("predict_macro_precision", self.predict_macro_precision, sync_dist=True)
        return {"predict_micro_precision": self.predict_micro_precision, "predict_macro_precision": self.predict_macro_precision}

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.resnet50.conv1) # type: ignore
        self.freeze(pl_module.resnet50.bn1) # type: ignore
    
    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: torch.optim.Optimizer) -> None:
        pass

def hot_to_cityscape_str(hot, num_classes):
    return ", ".join((labels.trainId2label[p.item()].name for p in hot.nonzero() if p < (num_classes - 1)))

