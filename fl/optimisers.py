import torch
from fl.models import NumpyModel
import numpy as np



class ClientSGD(torch.optim.SGD):
    """
    SGD on FL clients without momentum (hence ClientSGD has no parameters).
    """

    def __init__(self, params, lr):
        """
        Return a new ClientSGD optimizer.

        Args:
        - params:   {list} of torch tensors as returned by model.paramters()
        - lr:       {float} fixed learning rate to use
        """
        super(ClientSGD, self).__init__(params, lr)

    def get_params_numpy(self):
        """
        Returns a NumpyModel with 0 parameters.
        """
        return NumpyModel([])

    def set_params_numpy(self, params):
        pass
