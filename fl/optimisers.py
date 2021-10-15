import torch
from fl.models import NumpyModel


class FLClientOptimiser():
    """
    Used for Federated Averaging algorithms where the optimiser parameters
    need to be retrieved at set.
    """

    def get_params_numpy(self):
        """
        Returns NumpyModel containing copies of current optimiser parameters.
        """
        raise NotImplementedError()

    def set_params_numpy(self, params):
        """
        Set the optimizer parameters. Order of parameters in given NumpyModel
        must be the same as the order returned by get_params_numpy().

        Args:
        - params: {NumpyModel} containing new parameters
        """
        raise NotImplementedError()


class ClientSGD(torch.optim.SGD, FLClientOptimiser):
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


class ClientAdam(torch.optim.Adam, FLClientOptimiser):
    """
    Adam on FL clients. This Adam inherits from the Pytorch Adam implementation,
    so uses the dynamic learnign rate calculated using step, beta1, beta2.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-07, weight_decay=0):
        """
        Return a new ClientAdam optimizer.

        Args:
        - params:       {list} of torch tensors as returned by model.paramters()
        - lr            {float} initial learning rate to use
        - betas:        {tuple} of two floats for beta1, beta2 hyperparameters
        - eps:          {float} << 0, numerical stability constant
        - weight_decay: {float} 0 <= wd < 1, L2 regularisation parameter
        """
        super(ClientAdam, self).__init__(params, lr, betas, eps,
                                         weight_decay, amsgrad=False)

    def get_params_numpy(self):
        """
        Returns a NumpyModel containing iteration step count, average gradient
        and average squared gradient for each tensor in the tracked model.
        Therefore returns NumpyModel with n*3 values, where n is the number of
        parameters in the tracked model.

        Returns: NumpyModel
        """
        params = []
        for key in self.state.keys():
            params.append(self.state[key]['step'])
            params.append(self.state[key]['exp_avg'].cpu().numpy())
            params.append(self.state[key]['exp_avg_sq'].cpu().numpy())

        return NumpyModel(params)

    def set_params_numpy(self, params):
        """
        Set the parameters of this Adam optimizer. Order of parameters in the
        passed NumpyModel must be in the same order as those returned by
        ClientAdam.get_params_numpy().

        Args:
        - params: {NumpyModel} new parameters to set
        """
        i = 0
        for key in self.state.keys():
            self.state[key]['step'] = params[i]
            self.state[key]['exp_avg'].copy_(torch.tensor(params[i + 1]))
            self.state[key]['exp_avg_sq'].copy_(torch.tensor(params[i + 2]))
            i += 3
