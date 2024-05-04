# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class DefaultCollator(object):

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        return collated_batch, None, None
