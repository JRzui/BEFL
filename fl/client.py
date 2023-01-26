import numpy as np
from fl.models import NumpyModel
from fl.data_utils import PyTorchDataFeeder
from fl.reducer import Reducer
import torch

class Worker():
    def __init__(self, data_feeder, model, optimizer):
        self.data_feeder = data_feeder
        self.model = model
        self.model.set_optim(optimizer)
        self.reducer = None
        self.err = None

    def train_step(self, round_model, K, B):
        # perform local training
        self.data_feeder.activate()
        self.model.set_params(round_model)
        for k in range(K):
            x, y = self.data_feeder.next_batch(B)
            _, _ = self.model.train_step(x, y)
        self.data_feeder.deactivate()

    def get_grads(self, round_model):
        if type(round_model) != NumpyModel:
            round_model = NumpyModel(round_model)
        grads = round_model - self.model.get_params_numpy()
        if self.err != None:
            grads += self.err
        return grads

    def compress(self, grads, rank):
        if self.reducer == None:
            self.reducer = Reducer(rank)

        params = grads.params
        comp_grads, self.err = self.reducer.reduce(params)
        return comp_grads

    def upload(self, round_model, rank):
        grads = self.get_grads(round_model)
        return self.compress(grads, rank)

class Attacker():
    def attack(self, round_model, K, B):
        raise NotImplementedError()

    def attack_upload(self, round_model, rank):
        raise NotImplementedError()

class LF(Attacker, Worker):
    """
    The label filliping attack
    """
    def __init__(self, model, optimizer, data_feeder, dset_name):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_feeder.activate()
        x, y = data_feeder.all_x_data, data_feeder.all_y_data
        if dset_name == "mnist":
            poison_x = x.cpu().numpy()
            poison_y = 9 - y.cpu().numpy()
        elif dset_name == "femnist":
            poison_x = x.cpu().numpy()
            poison_y = 61 - y.cpu().numpy()
        elif dset_name == "cifar":
            poison_x = x.cpu().numpy()
            poison_y = 9 - y.cpu().numpy()
        else:
            raise ValueError("Incorrect dataset name.")
        adver_feeder = PyTorchDataFeeder(poison_x, torch.float32, poison_y, 'long', device)
        data_feeder.deactivate()

        super(LF, self).__init__(adver_feeder, model, optimizer)

    def attack(self,round_model, K, B):
        # perform adversarial local training
        self.train_step(round_model, K, B)

    def attack_upload(self, round_model, rank):
        return self.upload(round_model, rank)

class BF(Attacker, Worker):
    """
    The bit flipping attack
    """
    def __init__(self,data_feeder, model, optimizer):
        super(BF, self).__init__(data_feeder, model, optimizer)

    def attack(self, round_model, K, B):
        # perform normal local training
        self.train_step(round_model, K, B)

    def attack_upload(self, round_model, rank):
        grads = self.get_grads(round_model) * (-1)
        return self.compress(grads, rank)

