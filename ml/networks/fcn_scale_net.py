import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize as l2norm
import torch.nn.functional as F


class AdaptiveConcatPool2d(nn.Module):
    "Concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.ap = nn.AdaptiveAvgPool2d(self.k)
        self.mp = nn.AdaptiveMaxPool2d(self.k)

    def forward(self, x):
        ap = self.ap(x)
        mp = self.mp(x)
        return torch.cat([mp, ap], 1)


class SizeConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv11 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4
        )
        self.conv12 = nn.Conv2d(
            in_channels=96, out_channels=96, kernel_size=1, padding=0, stride=1
        )
        self.conv13 = nn.Conv2d(
            in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=2
        )
        self.bnorm1 = nn.BatchNorm2d(96)

        self.conv21 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1
        )
        self.conv22 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1
        )
        self.conv23 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=0, stride=2
        )
        # self.bnorm2 = nn.BatchNorm2d(256)

        self.conv31 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1
        )
        self.conv32 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=1, padding=0, stride=1
        )
        self.conv33 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, padding=0, stride=2
        )
        self.bnorm3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=1024, kernel_size=1, padding=0, stride=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1
        )

        self.dropout = nn.Dropout(0.5)

        self.global_avgpool = AdaptiveConcatPool2d()

    def forward(self, X):

        X = F.relu(self.conv11(X))
        X = F.relu(self.conv12(X))
        X = F.relu(self.conv13(X))
        X = self.bnorm1(X)
        X = self.dropout(X)

        X = F.relu(self.conv21(X))
        X = F.relu(self.conv22(X))
        X = F.relu(self.conv23(X))
        # X = self.bnorm2(X)
        X = self.dropout(X)

        X = F.relu(self.conv31(X))
        X = F.relu(self.conv32(X))
        X = F.relu(self.conv33(X))
        X = self.bnorm3(X)
        X = self.dropout(X)

        X = F.relu(self.conv4(X))
        X = self.dropout(X)

        X = self.conv5(X)
        # X = F.relu(X)

        X = self.global_avgpool(X)
        X = torch.flatten(X, start_dim=1)
        # X = self.ap(X)

        return X


class ScaleTripletModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_model = SizeConv()

    def forward_one(self, X):
        resize_input = self.resize_model(X)
        return resize_input

    def forward(self, q, p, n):

        Q = self.forward_one(q)
        P = self.forward_one(p)
        N = self.forward_one(n)

        return Q, P, N
