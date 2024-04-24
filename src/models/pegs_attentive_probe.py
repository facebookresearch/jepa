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
        self.linear = nn.Linear(12544*embed_dim, num_classes, bias=False)

    def forward(self, x):
        x = torch.sum(x, dim=1)
        x = self.linear(x)
        return x
        
        # print("input to classifier x shape:", x.shape)
        # flattened_x = x.flatten(1,-1)
        # print("flattened x:", flattened_x.shape)
        # x = self.linear(flattened_x)
