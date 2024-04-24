# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

    def forward(self, x):
        print("input to classifier shape:", x.shape)
        x = torch.sum(x, dim=1)
        print("summed x:", x.shape)
        x = self.linear(x)
        return x
        
        # flattened_x = x.flatten(1,-1)
        # print("flattened x:", flattened_x.shape)
        # x = self.linear(flattened_x)
