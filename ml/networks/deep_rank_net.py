import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.functional import normalize as l2norm


class AdaptiveConcatPool2d(nn.Module):
    "Concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.ap = nn.AdaptiveAvgPool2d(self.k)
        self.mp = nn.AdaptiveMaxPool2d(self.k)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class DeepRank(nn.Module):
    """
    Deep Image Rank Architecture
    """

    def __init__(self):
        super(DeepRank, self).__init__()

        self.resnet_model = models.resnet18(pretrained=True)
        # changing the adaptive avg pooling layer to combined average and max pooling
        self.resnet_model.avgpool = AdaptiveConcatPool2d()
        # self.conv_model.fc = nn.Identity()
        self.resnet_model.fc = nn.Linear(in_features=(1024), out_features=512)

    def forward_one(self, X):

        resnet_input = self.resnet_model(X)
        resnet_input = torch.flatten(resnet_input, start_dim=1)
        ### check scores with l2 norm ###
        # resnet_input = l2norm(resnet_input, p=2, dim=1)
        return resnet_input

    def forward(self, q, p, n):

        Q = self.forward_one(q)
        P = self.forward_one(p)
        N = self.forward_one(n)

        return Q, P, N
