import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # backbone
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # branch: displacement field
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_dp7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            Net.MeanShift(2)
        )

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.dp_layers = nn.ModuleList([self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7])

    class MeanShift(nn.BatchNorm2d):

        def __init__(self, num_features, momentum=0.1):
            super(Net.MeanShift, self).__init__(num_features, affine=False, momentum=momentum)

        def forward(self, input):

            if self.training:
                super(Net.MeanShift, self).forward(input).detach()
                return input

            return input - self.running_mean.view(1, 2, 1, 1)


    def forward(self, x):

        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_up = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        edge_out = edge_up

        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

        return edge_out, dp_out

    def trainable_parameters(self):

        return (tuple(self.edge_layers.parameters()), tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class AffinityDisplacement(Net):

    path_indices_prefix = "path_indices"

    def __init__(self, path_indices=None, ind_from=None, ind_to=None):

        super(AffinityDisplacement, self).__init__()

        self.n_path_type = len(path_indices)
        for i in range(self.n_path_type):
            param = torch.nn.Parameter(torch.from_numpy(path_indices[i]), requires_grad=False)
            self.register_parameter(AffinityDisplacement.path_indices_prefix + str(i), param)

        self.register_parameter(
            "ind_from",
            torch.nn.Parameter(torch.unsqueeze(ind_from, dim=0), requires_grad=False))
        self.register_parameter(
            "ind_to",
            torch.nn.Parameter(ind_to, requires_grad=False))


    def edge_to_affinity(self, edge):

        aff_list = []
        edge = edge.view(edge.size(0), -1)

        for i in range(self.n_path_type):
            ind = self._parameters[AffinityDisplacement.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)

        return aff_cat


    def forward(self, x):
        edge_out, dp_out = super().forward(x)
        edge_out = torch.sigmoid(edge_out)

        aff_out = self.edge_to_affinity(edge_out)

        return aff_out, dp_out


class EdgeDisplacement(Net):

    def __init__(self):
        super(EdgeDisplacement, self).__init__()

    def forward(self, x, out_settings=None):
        edge_out, dp_out = super().forward(x)

        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        dp_out = dp_out[0]

        return edge_out, dp_out


