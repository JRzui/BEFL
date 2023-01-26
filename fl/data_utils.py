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
from torchvision import datasets, transforms

    
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

def load_cifar10(num_users, n_class, n_samples, rate_unbalance):
    """
    This non-i.i.d cifar10 is implemented from
    https://github.com/jeremy313/non-iid-dataset-for-personalized-federated-learning
    the official implementation of the non-iid dataset in "LotteryFL: Personalized and
    Communication-Efficient Federated Learning with Lottery Ticket Hypothesis on Non-IID Datasets".
    Args:
    - num_users:        {int} the number of workers
    - n_class:          {int} the number of image classes each client has
    - n_samples:        {int{ the number of samples per class distributed to clients
    - rate_unbalance:   {float} unbalance rate of cifar10 data, (0-1) 1 denotes balanced
    """
    data_dir = '../data/CIFAR10_data/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)

    test_imgs, test_labels = test_dataset.data, test_dataset.targets
    train_imgs, train_labels = train_dataset.data, train_dataset.targets

    # transpose (rather than reshape) required to get actual order of
    # data in ndarray to change. If reshape is used, when data is
    # passed to a torchvision.transform, then the resulting images come
    # out incorrectly.
    train_imgs = [np.transpose(imgs, (2, 0, 1)) for imgs in train_imgs]
    test_imgs = [np.transpose(imgs, (2, 0, 1)) for imgs in test_imgs]

    # Chose euqal splits for every user
    num_shards_train, num_imgs_train = int(50000 / n_samples), n_samples
    num_classes = 10
    assert (n_class * num_users <= num_shards_train)
    assert (n_class <= num_classes)
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)

    labels = train_labels
    # labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]),
                                             axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]),
                    axis=0)
                user_labels = np.concatenate(
                    (user_labels, labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
            unbalance_flag = 1

    train_xs, train_ys = [], []
    train_imgs = np.array(train_imgs)
    train_labels = np.array(train_labels)
    for key in dict_users_train.keys():
        idx = dict_users_train[key].astype(int)
        train_x = train_imgs[idx]
        train_y = train_labels[idx]
        train_xs.append(train_x)
        train_ys.append(train_y)

    return (train_xs, train_ys), (test_imgs, test_labels)
    
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

def sum_model_L2_distance(x, y):
    """
    Args:
    - x: {NumpyModel}
    - y: {NumpyModel}

    Returns: {float} Sum L2 distance between tensors in x and y.
    """
    dists = (x - y) ** 2
    sums = [np.sum(d) for d in dists]
    sqrts = [np.sqrt(s) for s in sums]
    return np.sum(sqrts)
    
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
