import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        out0 = self.__forward(x)
        out1 = self.__forward(x.flip(-1))
        return out0 + out1.flip(-1)
    
    def __forward(self, x):
        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.classifier(x)
        x = F.relu(x)

        return x
