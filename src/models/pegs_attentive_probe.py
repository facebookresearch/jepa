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
        embed_dim=768,
        num_classes=165
    ):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        print("input to classifier x shape:", x.shape)
        x = self.linear(x)
        return x

