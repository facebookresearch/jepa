# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class PegAttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768, # note that the embed dim gets set from the encoder parameters (vit)
        num_classes=165
    ):
        super().__init__()
        # self.linear = nn.Linear(12544*embed_dim, num_classes, bias=False)
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)
        self.softmax = nn.Softmax()
        self.avgpool = nn.AvgPool2d(10, stride=2)

    def forward(self, x):
        print("input to classifier shape:", x.shape)
        x = torch.sum(x, dim=1)
        print("summed x:", x.shape)
        print("min after sum", torch.min(x))
        print("max after sum", torch.max(x))
        # x = self.softmax(x)
        # print("min after softmax", torch.min(x))
        # print("max after softmax", torch.max(x))
        x = self.avgpool(x)
        print("min after avgpool", torch.min(x))
        print("max after avgpool", torch.max(x))
        x = self.linear(x)
        print("min after linear", torch.min(x))
        print("max after linear", torch.max(x))
        return x
        
        # flattened_x = x.flatten(1,-1)
        # print("flattened x:", flattened_x.shape)
        # x = self.linear(flattened_x)
