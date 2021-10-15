from fl.data_utils import orthogonalize
from fl.models import NumpyModel

import numpy as np


class Reducer():
    def __init__(self, rank=1, reuse=False):
        self.rank = rank
        self.reuse = reuse
        self.qs = []

    def compress(self, M, q):
        p = np.matmul(M, q)  # Compute p
        p = orthogonalize(p)  # Orthogonalize p
        q = np.matmul(M.T, p)  # Compute q

        return p, q

    def decompress(self, p, q, param):
        return np.matmul(p, q.T).reshape(*param.shape)

    def reduce(self, params):
        """
        Reduce the size of gradients
        Args:
            - params:           {list of np.ndarray}
        Return:
             - comp_params:     {list} the compressed gradients
             - errs:            {list of np.ndarray} the error of original gradients and the decompressed one
        """
        comp_params = []
        errs = []
        idx = 0
        qs = []
        for param in params:
            if param.ndim <= 1:
                comp_params.append(param)  # rank1 array, do not need compress
                out = param
                errs.append(param - out)  # local error recording

            else:  # high rank array
                matrix = param.reshape(param.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)

                # Prepare q
                if self.qs == [] or self.reuse == False:
                    q = np.random.randn(m,  rank)  # randomly initialize q
                else:
                    q = self.qs[idx]

                p, q = self.compress(matrix, q)
                comp_params.append([p, q])
                qs.append(q) # update q
                idx += 1

                out = self.decompress(p, q, param)
                errs.append(param - out)  # local error recording

        self.qs = qs

        return comp_params, NumpyModel(errs)

