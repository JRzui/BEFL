import numpy as np
from fl.models import NumpyModel
from fl.data_utils import PyTorchDataFeeder
import torch
"""
def label_flipping(label, origin, target):
    
    This is targeted label flipping attack, changing the original label into the target one,

    Args:
        - label:        {torch.tensor} the labels of samples
        - origin:       {torch.dtype 'long'} the original label
        - target:       {torch.dtype 'long'} the targeted label

    No returns, since the tensor has already sent to device, the value of tensor changed
    This should be called after the feeder is active.
    
    idxs = label == origin
    label[idxs] = target # change the original label to the target
"""
def label_flipping(data_feeders, origin, target, device=torch.device('cuda:0')):
    adver_feeders = []
    for i in range(len(data_feeders)):
        data_feeders[i].activate()
        x, y = data_feeders[i].all_x_data, data_feeders[i].all_y_data
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        idxs = np.where(y == origin)
        poison_x = x[idxs]
        poison_y = np.ones(poison_x.shape[0]) * target
        adver_feeders.append(PyTorchDataFeeder(poison_x, torch.float32, poison_y, 'long', device))
        data_feeders[i].deactivate()
    return adver_feeders

def noise_gradient(params):
    """
    This is untargeted poisoning attack, client send arbitrary model updates.
    In this attack, we assume client send a random model updates that follow  normal (Gaussian) distribution

    Args:
        - params:           {list of np.ndarray} the model parameters of a NumpyModel

    Returns:
        - params_noise      {list of np.ndarray} the randomized gradients of a NumpyModel

    """
    grads_noise = []
    for param in params:
        grad_noise = np.random.standard_normal(size=param.shape)
        grads_noise.append(grad_noise)
    return NumpyModel(grads_noise)