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


class ClientSGDm(torch.optim.Optimizer):
    """
    SGD momentum on FL clients
    """

    def __init__(self, params, lr, beta, device):
        """
        Returns a new optimizer for use in mimelite Federated experiments.

        Args:
        - params:   {list} of params from a pytorch model
        - lr:       {float} learning rate
        - beta:     {float} momentum value, 0 <= beta < 1.0
        """
        defaults = dict(lr=lr, beta=beta)
        super(ClientSGDm, self).__init__(params, defaults)
        self.lr = lr
        self.beta = beta
        self.device = device
        self._init_m()

    def _init_m(self):
        """
        Initialise the momentum terms of the optimizer with zeros.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['m'] = torch.zeros_like(p,
                                              device=self.device,
                                              dtype=torch.float32)

    def step(self, closure=None):
        """
        Perform one step of momentum-SGD. As per Karimireddy et al. (Table 2,
        Appedix A), this optimizer does not update the momentum values at
        each step. $\mathcal{U}$ step in paper.

        Args:
        - closure: {callable} see torch.optim documentation

        Returns: {None, float} see torch.optim documentation
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data  # parameter gradient
                m = self.state[p]['m']  # current momentum value

                p.data.sub_(self.lr * (self.beta * m + (1 - self.beta) * g))

        return loss

    def update_moments(self, grads):
        """
        Update the momentum terms of the optimiser using the passed gradients,
        as per the rule in Table 2, Appendix A of Karimireddy et al 2020.
        $\mathcal{V}$ step in paper.

        Args:
        - grads: {NumpyModel} containing gradients to update with.
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                m = self.state[p]['m']
                # need to convert np.ndarray to torch.tensor
                g = torch.tensor(grads[i],
                                 dtype=torch.float32,
                                 device=self.device)

                self.state[p]['m'] = (self.beta * m) + ((1 - self.beta) * g)
                i += 1

    def set_params(self, moments):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # need to convert np.ndarray to torch.tensor
                m = torch.tensor(moments[i],
                                 dtype=torch.float32,
                                 device=self.device)

                self.state[p]['m'] = m
                i += 1

    def get_params_numpy(self):
        """
        Return momentum values of optimizer as a NumpyModel.
        """
        params = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:  # ignore gradient-less variables
                    continue

                params.append(np.copy(self.state[p]['m'].cpu().data.numpy()))

        return NumpyModel(params)

