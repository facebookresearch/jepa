# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Iterator, Optional
from operator import itemgetter
import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    Sampler,
    DistributedSampler,
    WeightedRandomSampler
)


class DatasetFromSampler(Dataset):

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """ Convert any Pytorch Sampler to a DistributedSampler """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """ Generalized WeightedRandomSampler to allow for more than 2^24 samples """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class DistributedWeightedSampler(DistributedSamplerWrapper):

    def __init__(
        self,
        weights,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        weighted_sampler = CustomWeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=False)

        super(DistributedWeightedSampler, self).__init__(
            sampler=weighted_sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
