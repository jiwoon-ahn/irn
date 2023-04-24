import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch
from torch import optim

import lightning
from lightning.pytorch.utilities import types

import typing

from torchmetrics.classification import MultilabelPrecision


class Net(lightning.LightningModule):
    

    def __init__(self, learning_rate: float, weight_decay: float, threshold: float):
        super(Net, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.precision = MultilabelPrecision(20, threshold=threshold)
        self.micro_precision = MultilabelPrecision(20, threshold=threshold, average='micro')
        self.macro_precision = MultilabelPrecision(20, threshold=threshold, average='macro')

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

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
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        return self

    def trainable_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad], [p for p in self.newly_added.parameters() if p.requires_grad]
    
    def on_train_start(self) -> None:
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        _, new_params = self.trainable_parameters()
        optimizer = optim.SGD(new_params, self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2, step_size_down=3, mode='triangular')
        return [optimizer], [scheduler]
    
    def training_step(self, batch, _) -> types.STEP_OUTPUT:
        img, hot_label = batch

        weights = torch.full((20, ), 1.0, device=self.device)
        for i in (0, 1, 2, 8):
            weights[i] = 0.05

        pred = self(img)
        loss = nn.functional.multilabel_soft_margin_loss(pred[:, :19], hot_label[:, :19], weights[:19])
        return loss
    
    def validation_step(self, batch, _) -> typing.Optional[types.STEP_OUTPUT]:
        (img, hot_label, unique_label) = batch
        unique_label = unique_label.to(dtype=torch.long)
        pred = self(img)
        loss = torch.nn.functional.multilabel_soft_margin_loss(pred, unique_label)
        self.log("val_loss", loss)

        self.precision.update(pred, hot_label)
        self.macro_precision.update(pred, hot_label)
        self.micro_precision.update(pred, hot_label)
    
        # if len(example_images) < num_examples:
        #     for i in range(min(num_examples - len(example_images), img.shape[0])):
        #         pred_str = hot_to_cityscape_str(thresholded_pred[i])
        #         label_str = hot_to_cityscape_str(hot_label[i])
        #         im = wandb.Image(img[i], caption=f"[prediction]: {pred_str}\n[label]: {label_str}")
        #         example_images.append(im)

    def on_validation_batch_end(self, outputs: typing.Optional[types.STEP_OUTPUT], batch: typing.Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

