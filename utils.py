import matplotlib.pyplot as plt
import itertools

def show_image(im):
    plt.imshow(im, interpolation='nearest')
    plt.show()

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