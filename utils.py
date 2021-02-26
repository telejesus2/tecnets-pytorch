import json
import os
import os.path as osp

def save_params(dir, params):
    if not(os.path.exists(dir)):
        os.makedirs(dir)
    with open(osp.join(dir, "params.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n','\t:\t'), sort_keys=False))

import matplotlib.pyplot as plt

def show_image(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()

import itertools

def chunked_iterable(iterable, size):
    # size must be of type int (not np.int64)
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

# DATASET = '/home/caor/Documents/datasets/demos/'
# from dataset import save_to_hdf5
# demos = np.load(DATASET + 'data0.npy', allow_pickle=True)
# d = demos[15:20]
# save_to_hdf5(d, 'ReachTarget')