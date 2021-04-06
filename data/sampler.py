import numpy as np
import torch
from utils import chunked_iterable

support_query_error = ('Tried to sample {} support and query samples, '
                       'but there are only {} samples of this subtask.')


class MetaSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, support_query_size, examples_size, shuffle=False, replacement=False):
        self.dataset = dataset
        self.support_query_size = support_query_size
        self.examples_size = examples_size
        self.shuffle = shuffle
        self.replacement = replacement

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for index in self._index_iterator():
            num_examples_of_subtask = self.dataset.len_subtask(index) 
            if num_examples_of_subtask < self.support_query_size:
                raise RuntimeError(support_query_error.format(self.support_query_size, num_examples_of_subtask))
            sample_indices = np.random.choice(num_examples_of_subtask, self.support_query_size, replace=False)
            ctrnet_timesteps = [4,5,6] #np.random.choice(self.dataset.time_horizon, self.examples_size, replace=False)
            yield [index, sample_indices, ctrnet_timesteps]

    def _index_iterator(self):
        n = len(self.dataset)
        if not self.shuffle:
            yield from range(n)
        else:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            if self.replacement:
                for _ in range(n // 32):
                    yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
                yield from torch.randint(high=n, size=(n % 32,), dtype=torch.int64, generator=generator).tolist()
            else:
                yield from torch.randperm(n, generator=generator).tolist()


# class BatchMetaSampler(torch.utils.data.sampler.Sampler):

#     def __init__(self, dataset, support_query_size, batch_size):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.support_query_size = support_query_size
#         self.num_subtasks = len(dataset.data)
#         self.samples_to_take = np.minimum(self.batch_size, self.num_subtasks)

#     def __iter__(self):
#         for subtask_indices in chunked_iterable(range(self.num_subtasks), int(self.samples_to_take)):
#             num_examples_per_subtask = [len(self.dataset.data[index]) for index in subtask_indices]
#             if np.max(num_examples_per_subtask) < self.support_query_size:
#                 raise RuntimeError(support_query_error.format(self.support_query_size, np.max(num_examples_per_subtask)))
#             sample_indices = [np.random.choice(num_examples_per_subtask[i], self.support_query_size, replace=False) for i in range(len(subtask_indices))]
#             yield [np.concatenate(([x], y)) for x, y in zip(subtask_indices, sample_indices)]