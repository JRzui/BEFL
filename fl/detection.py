import torch
from math import log
import numpy as np
from fl.models import NumpyModel

class Detect():
    def __init__(self, clients_grads, global_model):
        """
        Args:
        - client_grads：             {list of the list of np.ndarray} the grads of model parameters of a NumpyModel, each contains a client's model updates
        - global_model:              {list of np.ndarray} the global model parameters of a NumpyModel
        """
        self.clients_grads = clients_grads
        self.global_model = NumpyModel(global_model)
        self.clients_model = [self.global_model - NumpyModel(client_grads) for client_grads in clients_grads]


    def get_global_mi(self, unlabeled_data, model):
        """
        Get the mutual information value between client model and global model

        Args:
            - unlabeled_data:           {list of tensors} the unlabeled dataset, only contains x, no y included
            - model:                    {FLModel}

        Returns:
            - MI:                       (np.ndarray} the corresponding MI value between client model and golobal model
        """

        model.set_params(self.global_model)
        global_out = model.forward(unlabeled_data) # global model output
        global_exp = global_out.mean(dim=1, keepdim=True) # the expectation of global model

        # calculate the mutual information between client model and global model
        MI = []
        for client_model in self.clients_model:
            model.set_params(client_model)
            client_out = model.forward(unlabeled_data) # client model output
            client_exp = client_out.mean(dim=1, keepdim=True) # the expectation of client model

            rho = torch.sum((global_out - global_exp) * (client_out - client_exp), dim=1) / \
                  torch.sqrt(torch.sum(torch.square(global_out - global_exp), dim=1) * \
                             torch.sum(torch.square(client_out - client_exp), dim=1))

            intermediate = 1 - pow(rho.mean().item(), 2)
            # prevent math domain error
            if intermediate == 0:
                intermediate += 1e-9
            mi = - log(intermediate) / 2
            MI.append(mi)
        return np.array(MI)

    def get_mutual_mi(self, unlabeled_data, model):
        """
        Get the mutual information value between client model and global model

        Args:
            - unlabeled_data:           {list of tensors} the unlabeled dataset, only contains x, no y included
            - model:                    {FLModel}

        Returns:
            - MI:                       (np.ndarray} the corresponding MI value between client model and golobal model
        """
        mutual_mi = np.zeros(shape=(len(self.clients_model), len(self.clients_model)))
        for i in range(len(self.clients_model)):
            model.set_params(self.clients_model[i])
            client_out_i = model.forward(unlabeled_data)
            client_exp_i = client_out_i.mean(dim=1, keepdim=True) # the expectation of model output
            for j in range(i+1, len(self.clients_model)):
                model.set_params(self.clients_model[j])
                client_out_j = model.forward(unlabeled_data)
                client_exp_j = client_out_j.mean(dim=1, keepdim=True)  # the expectation of model output

                rho = torch.sum((client_out_i - client_exp_i) * (client_out_j - client_exp_j), dim=1) / \
                      torch.sqrt(torch.sum(torch.square(client_out_i - client_exp_i), dim=1) * \
                                 torch.sum(torch.square(client_out_j - client_exp_j), dim=1))

                intermediate = 1 - pow(rho.mean().item(), 2)
                # prevent math domain error
                if intermediate == 0:
                    intermediate += 1e-9
                mi = - log(intermediate) / 2

                mutual_mi[i, j] = mi
                mutual_mi[j, i] = mi
        avg_mi = np.mean(mutual_mi, axis=1)
        return avg_mi

    def detect(self, unlabeled_data, model):
        """
        Detect the potential malicious model updates according to the robust estimator MAD and MADN and
        the traditional "three-sigma edit" rule

        Args:
            - unlabeled_data:           {list of tensors} the unlabeled dataset, only contains x, no y included
            - model:                    {FLModel}

        Returns:
            - select:               {list of the list of np.ndarray} the selected clients model updates
        """
        select = []
        """
        # fuzzy-logic based outlier detection
        self.MI = self.get_mutual_info(unlabeled_data, model)  # get MI values

        centralMI = np.mean(self.MI)
        dist = abs(self.MI - centralMI) # the distance from centroid
        n_point = find_neighbours(self.MI, r=2) # number of neighbouring points
        """

        # Median based method
        mutual_mis = self.get_mutual_mi(unlabeled_data, model)
        global_mis = self.get_global_mi(unlabeled_data, model)
        MI = mutual_mis + global_mis
        
        MAD = np.median(abs(MI - np.median(MI))) # get the median absolute deviation from median of MI values
        MADN = MAD / 0.6745 # get the normalized MAD values, note 0.6745 is the MAD of a standard normal distribution

        # select client models, according to the "three-sigma edit" rule
        ts = (MI - np.median(MI)) / MADN
        for i in range(len(ts)):
            if abs(ts[i]) < 3:
                select.append(self.clients_grads[i])

        """
        # Trimmed mean based method
        beta = 0.05
        idx_sort = np.argsort(self.MI)
        low = int(len(self.MI) * beta)
        high = len(self.MI) - low

        # Remove the beta portion of the smallest and the largest MI
        for i in range(low,high):
            select.append(self.clients_grads[idx_sort[i]])
        """

        return select


