import numpy as np
import torch
import logging
from data.sampler import MetaSampler
from torch.utils.data import DataLoader

class MetaDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, support_size, query_size, examples_size, shuffle=True, replacement=False,
                 batch_size=1, num_workers=0):

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

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers)
