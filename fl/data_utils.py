"""
Functions for loading FEMNIST, CIFAR100, Shakespeare, StackOverflow datasets.
PyTorchDataFeeder class for conveniently containing a single worker's dataset,
and retrieving a stream of data batches from it. Also has some data utility 
functions.
"""
import torch

import pickle
import numpy as np
import os
import json
import h5py
import scipy.sparse
import idx2numpy
from fl.models import *

    
class PyTorchDataFeeder():
    """
    Used to easily contain the samples of a FL worker. Can hold the samples on 
    the GPU, and produce an endless stream of randomly drawn samples with a 
    given transformation applied.
    """

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    cast_device=None, transform=None):
        """
        Return a new PyTorchDataFeeder with copies of x and y as torch.tensors.
        Data will be stored on device. If x_dtype or y_dtype are the string 
        'long' then these tensors will be cast to the torch long dtype (used
        typically when pytorch models are expecting integer values). If 
        cast_device is passed, the data returned by next_batch will be cast to 
        this device. Doing so allows data held on the CPU to be easily fed to a 
        model sitting in the GPU memory, if, for example, all data won't fit in
        GPU memory. If transform is passed, the samples returned by next_batch 
        are transformed by this function.
       
        Args:
        - x:            {numpy.ndarray, torch.tensor} of samples
        - x_dtype:      {torch.dtype, 'long'} that x will be
        - y:            {numpy.ndarray, torch.tensor} of targets
        - y_dtype:      {torch.dtype, 'long'} that y will be 
        - device:       {torch.device} that x and y will sit on
        - cast_device:  {torch.device} next_batch returned data will be on here
        - transform:    {callable} applied by next_batch
        """
        self.x, self.x_sparse = self._matrix_type_to_tensor(x, x_dtype, device)
        self.y, self.y_sparse = self._matrix_type_to_tensor(y, y_dtype, device)
        self.idx              = 0
        self.n_samples        = x.shape[0]
        self.cast_device      = cast_device
        self.transform        = transform
        self.active           = False
        self.activate()
        self._shuffle_data()
        self.deactivate()
        
    def _matrix_type_to_tensor(self, matrix, dtype, device):
        """
        Converts a scipy.sparse.coo_matrix or a numpy.ndarray into a 
        torch.sparse_coo_tensor or torch.tensor. 
        
        Args:
        - matrix:   {scipy.sparse.coo_matrix or np.ndarray} to convert
        - dtype:    {torch.dtype} of the tensor to make 
        - device:   {torch.device} where the tensor should be placed
        
        Returns: (tensor, is_sparse)
        - tensor:       {torch.sparse_coo_tensor or torch.tensor}
        - is_sparse:    {bool} True if returning a torch.sparse_coo_tensor
        """
        if type(matrix) == scipy.sparse.coo_matrix:
            is_sparse = True
            idxs = np.vstack((matrix.row, matrix.col))

            if dtype == 'long':
                tensor = torch.sparse_coo_tensor(   idxs, 
                                                    matrix.data, 
                                                    matrix.shape, 
                                                    device=device, 
                                                    dtype=torch.int32).long()
                                
            else:
                tensor = torch.sparse_coo_tensor(   idxs, 
                                                    matrix.data, 
                                                    matrix.shape, 
                                                    device=device, 
                                                    dtype=dtype)
        
        elif type(matrix) == np.ndarray:
            is_sparse = False
            if dtype == 'long':
                tensor = torch.tensor(  matrix, 
                                        device=device, 
                                        dtype=torch.int32).long()
            else:
                tensor = torch.tensor(  matrix, 
                                        device=device, 
                                        dtype=dtype)
        else:
            raise TypeError('Only np.ndarray/scipy.sparse.coo_matrix accepted.')
        
        return tensor, is_sparse
        
        
        
    def activate(self):
        """
        Activate this PyTorchDataFeeder to allow .next_batch(...) to be called. 
        Will turn torch.sparse_coo_tensors into dense representations ready for 
        training.
        """
        self.active = True
        self.all_x_data = self.x.to_dense() if self.x_sparse else self.x
        self.all_y_data = self.y.to_dense() if self.y_sparse else self.y
        
    def deactivate(self):
        """
        Deactivate this PyTorchDataFeeder to disallow .next_batch(...). Will 
        deallocate the dense matrices created by activate to save memory.
        """
        self.active = False
        self.all_x_data = None
        self.all_y_data = None
       
        
    def _shuffle_data(self):
        """
        Co-shuffle the x and y data.
        """
        if not self.active:
            raise RuntimeError('_shuffle_data(...) called when feeder not active.')
        
        ord     = torch.randperm(self.n_samples)
        self.x = self.all_x_data[ord].to_sparse() if self.x_sparse else self.all_x_data[ord]
        self.y = self.all_y_data[ord].to_sparse() if self.y_sparse else self.all_y_data[ord]
        
    def next_batch(self, B):
        """
        Return a batch of randomly ordered data from this dataset. If B=-1, 
        return all the data as one big batch. If self.cast_device is not None, 
        then data will be sent to this device before being returned. If 
        self.transform is not None, that function will be applied to the data 
        before being returned.
        
        Args:
        - B: {int} size of batch to return.
        """
        if not self.active:
            raise RuntimeError('next_batch(...) called when feeder not active.')
        
        if B == -1:                 # return all data as big batch
            x = self.all_x_data
            y = self.all_y_data
            self._shuffle_data()
            
        elif self.idx + B > self.n_samples: # need to wraparound dataset 
            extra       = (self.idx + B) - self.n_samples
            x           = torch.cat((   self.all_x_data[self.idx:], 
                                        self.all_x_data[:extra]))
            y           = torch.cat((   self.all_y_data[self.idx:], 
                                        self.all_y_data[:extra]))
            self._shuffle_data()
            self.idx    = 0
            
        else:   # next batch can easily be obtained
            x           = self.all_x_data[self.idx:self.idx+B]
            y           = self.all_y_data[self.idx:self.idx+B]
            self.idx    += B
            
        if not self.cast_device is None:        # send to cast_device
            x           = x.to(self.cast_device)
            y           = y.to(self.cast_device)
    
        if not self.transform is None:          # perform transformation
            x           = self.transform(x)

        return x, y


def load_mnist(data_dir, W, P=0.5):
    """
    Load the MNIST data from file directory

    Args:
    - data_dir:    {string} the path that contains data files
    - W:           {int} number of workers' worth of data to load
    - P:           {float} the probability of non-i.i.d

    Returns: (train_x, train_y), (test_x, test_y)
    - train_xs: {list} of np.ndarrays
    - train_ys: {list} of np.ndarrays
    - test_x:  {np.ndarray}
    - test_y:  {np.ndarray}
    """
    train_x = idx2numpy.convert_from_file(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1) / 255
    train_x = np.transpose(train_x, (0, 3, 1, 2))
    train_y = idx2numpy.convert_from_file(os.path.join(data_dir, 'train-labels-idx1-ubyte'))

    test_x = idx2numpy.convert_from_file(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1) / 255
    test_x = np.transpose(test_x, (0, 3, 1, 2))
    test_y = idx2numpy.convert_from_file(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    # Partition training samples by labels
    train_xs = {}
    group_x = {}
    group_y = {}
    for label in range(10):
        idxs = np.where(train_y == label)
        train_xs[label] = train_x[idxs]
        group_x[label] = []
        group_y[label] = []

    # Assign samples to each group
    for l in range(10):
        n_samples_label = len(train_xs[l])
        probs = np.random.uniform(size=n_samples_label)

        group_x[l].append(train_xs[l][probs <= P])
        group_y[l].append(np.ones(int(sum(probs <= P))) * l)

        group_other_x = train_xs[l][probs > P]
        group_other_y = np.ones(int(sum(probs > P))) * l
        n_samples_each = int(len(group_other_x) / 9)
        i = 0
        for other in range(10):
            if other != l: # the remaining groups
                group_x[other].append(group_other_x[i * n_samples_each:(i+1) * n_samples_each])
                group_y[other].append(group_other_y[i * n_samples_each:(i+1) * n_samples_each])
                i += 1

    # Assign samples to each worker
    group_workers = int(W / 10)
    train_xs, train_ys = [], []
    for l in range(10):
        group_x[l] = np.concatenate(group_x[l])
        group_y[l] = np.concatenate(group_y[l])

        # Shuffle the training samples in the group
        idxs = np.random.permutation(len(group_x[l]))
        group_x[l] = group_x[l][idxs]
        group_y[l] = group_y[l][idxs]

        n_samples = int(len(group_x[l]) / group_workers)
        for i in range(group_workers):
            train_x_ = group_x[l][i * n_samples:(i + 1) * n_samples]
            train_y_ = group_y[l][i * n_samples:(i + 1) * n_samples]

            train_xs.append(train_x_)
            train_ys.append(train_y_)

    return (train_xs, train_ys), (test_x, test_y)



def load_femnist(train_dir, test_dir, W):
    """
    Load the FEMNIST data contained in train_dir and test_dir. These dirs should
    contain only .json files that have been produced by the LEAF 
    (https://leaf.cmu.edu/) preprocessing tool. Will load W workers' worth of 
    data from these files.
    
    Args:
    - train_dir:    {str} path to training data folder
    - test_dir:     {str} path to test data folder
    - W:            {int} number of workers' worth of data to load
    
    Returns: (x_train, y_train), (x_test, y_test)
    - x_train: {list} of np.ndarrays
    - y_train: {list} of np.ndarrays
    - x_test:  {np.ndarray}
    - y_test:  {np.ndarray}
    """
    train_fnames    = sorted([train_dir+'/'+f for f in os.listdir(train_dir)])
    test_fnames     = sorted([test_dir+'/'+f for f in os.listdir(test_dir)])
    # each .json file contains data for 100 workers
    n_files         = int(np.ceil(W / 100))
    
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []

    tot_w = 0
    for n in range(n_files):
        with open(train_fnames[n], 'r') as f:
            train = json.load(f)
        with open(test_fnames[n], 'r') as f:
            test = json.load(f)
        
        keys = sorted(train['user_data'].keys())
    
        for key in keys:
            # (1.0 - data) so images are white on black like classic MNIST
            x = 1.0 - np.array(train['user_data'][key]['x'], dtype=np.float32)
            x = x.reshape((x.shape[0], 28, 28, 1))
            # transpose (rather than reshape) required to get actual order of 
            # data in ndarray to change. If reshape is used, when data is 
            # passed to a torchvision.transform, then the resulting images come
            # out incorrectly.
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.array(train['user_data'][key]['y'])
            
            x_train.append(x)
            y_train.append(y)
            
            x = 1.0 - np.array(test['user_data'][key]['x'], dtype=np.float32)
            x = x.reshape((x.shape[0], 28, 28, 1))
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.array(test['user_data'][key]['y'])
            
            x_test.append(x)
            y_test.append(y)
            
            tot_w += 1
            
            if tot_w == W:
                break
                
    assert tot_w == W, 'Could not load enough workers from files.'
    
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return (x_train, y_train), (x_test, y_test)
    
    
    
def load_stackoverflow_lr(  train_file, test_file, W, n_test,
                            max_user_samples=500):
    """
    Load the StackOverflow data contained in train_file and test_file. The 
    train_file should contain a tuple of length 2, the first a list of training 
    x data as scipy.sparse.coo_matrix's, the second a list of training y data 
    as scipy.sparse.coo_matrix's. Test file contains a tuple of two coo_matrix's
    for x and y. Will load W arrays from train_file, and will limit each 
    worker's data to max_user_samples. Will load n_test samples from test_file.
    
    Args:
    - train_file:       {str} path to training data file
    - test_file:        {str} path to test data file
    - W:                {int} number of workers' worth of data to load
    - n_test:           {int} number of test samples to load
    - max_user_samples: {int} worker datasets will be capped at this
    
    Returns: (user_xs, user_ys), (test_xs, test_ys)
    - user_xs: {list} of scipy.sparse.coo_matrix
    - user_ys: {list} of scipy.sprase.coo_matrix
    - test_xs: np.ndarray
    - test_ys: np.ndarray
    """
    with open(train_file, 'rb') as f:
        user_xs, user_ys = pickle.load(f)
    
    with open(test_file, 'rb') as f:
        test_xs, test_ys = pickle.load(f)
    
    # convert to standard array if scipy sparse representation
    user_xs = user_xs[:W]
    user_ys = user_ys[:W]
    
    # choose a random set of testing samples 
    test_idxs = np.random.permutation(test_xs.get_shape()[0])[:n_test]
    test_xs = test_xs.toarray()[test_idxs,:]
    test_ys = test_ys.toarray()[test_idxs,:]
    
    return (user_xs, user_ys), (test_xs, test_ys)



def load_shakes(train_fname, test_fname, W):
    """
    Load the Shakespeare data contained in train_fname and test_fname. These 
    files should be .json files that have been produced by the LEAF 
    (https://leaf.cmu.edu/) preprocessing tool. Will load W workers' worth of 
    data from these files.
    
    Args:
    - train_fname: {str} path to training data file
    - test_fname:  {str} path to test data file
    - W:           {int} number of workers' worth of data to load
    
    Returns: (train_x, train_y), (test_x, test_y)
    - train_x: {list} of np.ndarrays
    - train_y: {list} of np.ndarrays
    - test_x:  {np.ndarray}
    - test_y:  {np.ndarray}
    """
    if W > 660:
        raise ValueError('Shakespeare dataset has max 660 users.')
    
    # all the symbols in the shakespeare text
    vocab       = ' !"&\'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}'
    chr_to_int  = {c:i for (i, c) in enumerate(vocab)}
    
    with open(train_fname, 'r') as f:
        train = json.load(f)
    with open(test_fname, 'r') as f:
        test = json.load(f)
    
    train_x, train_y, test_x, test_y = [], [], [], []
    
    users = sorted(train['users'])
    
    # load W worth of character text, and convert to ints
    for w in range(W):
        for (source, dest) in zip([train, test], [train_x, test_x]):
            x = source['user_data'][users[w]]['x']
            x = [[chr_to_int[c] for c in sentence] for sentence in x]
            dest.append(np.array(x, dtype=np.int32))
        
        for (source, dest) in zip([train, test], [train_y, test_y]):
            y = source['user_data'][users[w]]['y']
            y = [chr_to_int[c] for c in y]
            dest.append(np.array(y, dtype=np.int32))
        
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    
    return (train_x, train_y), (test_x, test_y)



def normalise(x, means, stds):
    """
    Centers images in x using means, and divides by stds. 
    
    Args:
    - x:        {np.ndarray} of shape [n_imgs, 3, img_h, img_w]
    - means:    {list} of floats containing per-channel means
    - stds:     {list} of floats containing per-channel standard deviations
    
    Returns:
    {np.ndarray} of x, normalised per channel. 
    """
    x1 = np.copy(x)
    for i in range(3):
        x1[:,i,:,:] = (x1[:,i,:,:] - means[i]) / stds[i]

    return x1



def load_cifar100(train_fname, test_fname, W):
    """
    Load the CIFAR100 data contained in train_fname and test_fname. These 
    files should be .h5py files downloaded by tensorflow federated when using
    the CIFAR100 dataset function.
    
    Args:
    - train_fname: {str} path to training data file
    - test_fname:  {str} path to test data file
    - W:           {int} number of workers' worth of data to load
    
    Returns: (train_x, train_y), (test_x, test_y)
    - train_imgs:   {list} of np.ndarrays
    - train_labels: {list} of np.ndarrays
    - test_imgs:    {np.ndarray}
    - test_labels:  {np.ndarray}
    """
    with h5py.File(train_fname, 'r') as f:
        keys            = sorted(list(f['examples'].keys()))[:W]
        train_imgs      = [f['examples'][k]['image'][()]/255.0 for k in keys]
        train_labels    = [f['examples'][k]['label'][()] for k in keys]
    
    with h5py.File(test_fname, 'r') as f:
        keys            = sorted(list(f['examples'].keys()))
        test_imgs       = [f['examples'][k]['image'][()]/255.0 for k in keys]
        test_labels     = [f['examples'][k]['label'][()] for k in keys]
    
    # transpose (rather than reshape) required to get actual order of 
    # data in ndarray to change. If reshape is used, when data is 
    # passed to a torchvision.transform, then the resulting images come
    # out incorrectly.
    train_imgs  = [np.transpose(imgs, (0, 3, 1, 2)) for imgs in train_imgs]
    test_imgs   = [np.transpose(imgs, (0, 3, 1, 2)) for imgs in test_imgs]
    
    test_imgs   = np.concatenate(test_imgs)
    test_labels = np.concatenate(test_labels)
    
    # means and stds computed per channel for entire dataset
    means   = [0.4914, 0.4822, 0.4465]
    stds    = [0.2023, 0.1994, 0.2010]
    
    for w in range(W):
        train_imgs[w] = normalise(train_imgs[w], means, stds)
    
    test_imgs = normalise(test_imgs, means, stds)
    
    return (train_imgs, train_labels), (test_imgs, test_labels)
    
def to_tensor(x, device, dtype):
    """
    Returns x as a torch.tensor.
    
    Args:
    - x:      {np.ndarray} data to convert
    - device: {torch.device} where to store the tensor
    - dtype:  {torch.dtype or 'long'} type of data
    
    Returns: {torch.tensor}
    """
    if dtype == 'long':
        return torch.tensor(x, device=device, 
                            requires_grad=False, dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device, requires_grad=False, dtype=dtype)



def step_values(x, m):
    """
    Return a stepwise copy of x, where the values of x that are equal to m are 
    taken from the last non-m value of x.
    
    Args:
    - x: {np.ndarray} values to make step-wise
    - m: {number} (same type as x) value to step over/ignore
    """
    stepped = np.zeros_like(x)
    curr = x[0]
    
    for i in range(1, x.size):
        if x[i] != m:
            curr = x[i]
        stepped[i] = curr
    
    return stepped



def avg_model_L1_distance(x, y):
    """
    Args:
    - x: {NumpyModel}
    - y: {NumpyModel} 
    
    Returns: {float} Average L1 distance between tensors in x and y.
    """
    dists   = (x - y).abs()
    sums    = [np.sum(d) for d in dists]
    return np.mean(sums)
    
    
def avg_model_L2_distance(x, y):
    """
    Args:
    - x: {NumpyModel}
    - y: {NumpyModel} 
    
    Returns: {float} Average L2 distance between tensors in x and y.
    """
    dists   = (x - y) ** 2
    sums    = [np.sum(d) for d in dists]
    sqrts   = [np.sqrt(s) for s in sums]
    return np.mean(sqrts)
    
    
def avg_model_cosine_angle(x, y):
    """
    Args:
    - x: {NumpyModel}
    - y: {NumpyModel}
    
    Returns {float} Average cosine angle between tensors in x and y.
    """
    cosines = []
    for (x_i, y_i) in zip(x, y):
        x_flt = x_i.flatten()
        y_flt = y_i.flatten()
        cos = np.dot(x_flt, y_flt) / (np.sqrt(x_flt.dot(x_flt)) * np.sqrt(y_flt.dot(y_flt)))
        cosines.append(cos)
        
    return np.mean(cosines)
    
def aggregate(grads, round_agg):
    """
    Args:
        - grads: {list of list of ndarrys}

    Returns:
         - round_agg: {NumpyModel}
    """

    for grad in grads:
        round_agg += grad

    return round_agg

def n_bits(array):
    """
    Args:
        - array:    {np.ndarray}

    Returns:
        - bits:     {int} the bits of the array
    """
    bits = 8 * array.nbytes
    return bits

def orthogonalize(matrix, eps=1e-8):
    n, m  = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i:i+1]
        col /= np.sqrt(np.sum(col ** 2)) + eps
        matrix[:, i:i+1] = col
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i+1:]
            rest -= np.sum(col * rest, axis=0) * col
            matrix[:, i+1:] = rest
    return matrix
