import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.utilities import types

import typing

from torchmetrics.classification import MultilabelPrecision
import cityscapesscripts.helpers.labels as labels
import wandb

class Net(pl.LightningModule):
    num_classes = 20
    num_examples = 8
    def __init__(self, learning_rate: float, weight_decay: float, threshold: float):
        super(Net, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.precision = MultilabelPrecision(self.num_classes, threshold=threshold)
        self.micro_precision = MultilabelPrecision(self.num_classes, threshold=threshold, average='micro')
        self.macro_precision = MultilabelPrecision(self.num_classes, threshold=threshold, average='macro')

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

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)
        return x    

    def train(self, mode=True):
        super().train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        return self
    
    def newly_added_params(self) -> typing.List[torch.nn.parameter.Parameter]:
        return [p for p in self.newly_added.parameters() if p.requires_grad]

    def on_train_start(self) -> None:
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def configure_optimizers(self) -> typing.Tuple[typing.List[torch.optim.Optimizer], typing.List[torch.optim.lr_scheduler.LRScheduler]]:
        new_params = self.newly_added_params()
        optimizer = optim.SGD(new_params, self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2, step_size_down=3, mode='triangular')

        return [optimizer], [scheduler]
    
    def training_step(self, batch, _) -> types.STEP_OUTPUT:
        _, _, _, img, hot_label = batch
        pred = self(img)
        loss = self.training_loss(pred[:, :(self.num_classes - 1)], hot_label[:, :(self.num_classes - 1)])
        self.log("train/loss", loss)
        
        return loss
    
    def on_train_batch_end(self, outputs: types.STEP_OUTPUT, batch: typing.Any, batch_idx: int) -> None:
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    # def validation_step(self, batch, _) -> typing.Optional[types.STEP_OUTPUT]:
    #     (img, hot_label, unique_label) = batch
    #     unique_label = unique_label.to(dtype=torch.long)
    #     pred = self(img)
    #     loss = self.validation_loss(pred, unique_label)

    #     self.precision.update(pred, hot_label)
    #     self.macro_precision.update(pred, hot_label)
    #     self.micro_precision.update(pred, hot_label)

    #     thresholded_pred = (pred > 0.5).to(dtype=torch.long)
    
    #     example_images = []
    #     if len(example_images) < self.num_examples:
    #         for i in range(min(self.num_examples - len(example_images), img.shape[0])):
    #             pred_str = hot_to_cityscape_str(thresholded_pred[i], self.num_classes)
    #             label_str = hot_to_cityscape_str(hot_label[i], self.num_classes)
    #             im = wandb.Image(img[i], caption=f"[prediction]: {pred_str}\n[label]: {label_str}")
    #             example_images.append(im)
        
    #     wandb.log(
    #         "example_images", example_images,
    #         "val/precision", self.precision,
    #         "val/macro_precision", self.macro_precision,
    #         "val/micro_precision", self.micro_precision,
    #         "val/loss", loss,
    #     )
    #     return {"val_loss" : loss, "precision": self.precision, "macro_precision": self.macro_precision, "micro_precision": self.micro_precision}

    # def on_validation_batch_end(self, outputs: typing.Optional[types.STEP_OUTPUT], batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    #     return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    # def on_validation_epoch_end(self) -> None:
    #     wandb.log({
    #         "val/precision": self.precision.compute(), 
    #         "val/macro_precision": self.macro_precision.compute(), 
    #         "val/micro_precision": self.micro_precision.compute()
    #     })
    #     self.precision.reset()
    #     self.macro_precision.reset()
    #     self.micro_precision.reset()    

def hot_to_cityscape_str(hot, num_classes):
    return ", ".join((labels.trainId2label[p.item()].name for p in hot.nonzero() if p < (num_classes - 1)))

