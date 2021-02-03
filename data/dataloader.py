import numpy as np
import torch
import logging
from data.sampler import MetaSampler
from torch.utils.data import DataLoader


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(DataLoader):
    """ Removes overhead from reinitializing workers after each epoch """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._DataLoader__initialized = False
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        # self._DataLoader__initialized = True
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class MetaDataLoader(FastDataLoader):

    def __init__(self, dataset, support_size, query_size, examples_size, shuffle=True, replacement=False,
                 batch_size=1, num_workers=0, prefetch_factor=2, pin_memory=False):

        self.support_size = support_size
        self.query_size = query_size
        self.examples_size = examples_size

        num_subtasks = len(dataset)
        samples_to_take = np.minimum(batch_size, num_subtasks)
        if samples_to_take != batch_size:
            logging.warning('Batch size was greater than number of subtasks.')

        if examples_size % 2 == 1:
            raise ValueError('Examples should be an even number.')

        sampler = MetaSampler(dataset, support_size + query_size, examples_size, shuffle, replacement)

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory)




