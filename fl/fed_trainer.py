from progressbar import progressbar
from fl.attack import *
from fl.detection import *
from fl.data_utils import aggregate
from fl.reducer import Reducer
import numpy as np

# fedavg with adversaries
def run_fedavg_adver(attack, data_feeders, test_data, model,
                     T, K, B, C, Cm, rank, seed, test_freq=1, test_B=64, detection=False, compress=False):
    """
    Args:
    - attack:       {string} the attack type, 'label-flipping' or 'arbitrary'
    - data_feeders: {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - T:            {int} number of rounds of FL
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - C:            {float} the fraction of participated clients in each round
    - Cm:           {float} the fraction of malicious clients existed in each round if the adversaries is set to True
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    - detection:    {bool} the mode of detection
    - compress:     {bool} the mode of compression

    Returns: test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then
    the test arrays will have 0s in the non-tested rounds.
    """
    np.random.seed(seed)

    test_errs = np.zeros((T), dtype=np.float32)
    test_accs = np.zeros((T), dtype=np.float32)

    unlabeled_size = 100
    unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]

    round_model = model.get_params_numpy()  # current global model
    round_agg = model.get_params_numpy()  # client aggregate model

    num_usr = data_feeders.__len__()
    num_adver = int(num_usr * C * Cm)  # the number of adversaries in each round
    num_honest = int(num_usr * C - num_adver)  # the number of honest worker in each round

    if compress == True:
        errs = {}
        # momentum setting
        lr = 1
        beta = 0.9
        m = round_model.zeros_like()

    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        # id's of honest workers participating in this round
        user_idxs = np.random.choice(num_usr - num_adver, num_honest, replace=False) + num_adver

        client_grads = []

        # honest workers training
        for user_idx in user_idxs:
            model.set_params(round_model)  # 'download' global model

            # perform local training
            data_feeders[user_idx].activate()
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                _, _ = model.train_step(x, y)
            data_feeders[user_idx].deactivate()

            # 'upload' model deltas
            client_model = model.get_params_numpy()
            client_grad = round_model - client_model

            if compress == True:
                if user_idx in errs.keys():
                    client_grad += errs[user_idx]  # error feedback
                reducer = Reducer(rank, reuse=True)
                comp_grad, err = reducer.reduce(client_grad.params)  # 'upload' compress gradients
                errs[user_idx] = err  # local error

                # server decompress gradients
                decomp_grad = []
                for i in range(len(comp_grad)):
                    if type(comp_grad[i]) == list:  # grads in layer i that being compressed
                        p = comp_grad[i][0]
                        q = comp_grad[i][1]
                        decomp_grad.append(reducer.decompress(p, q, round_model.params[i]))
                    else:
                        decomp_grad.append(comp_grad[i])
                client_grad = NumpyModel(decomp_grad)

            client_grads.append(client_grad)

        # adversaries training
        if attack == 'label-flipping':
            adver_feeders = data_feeders[:num_adver]   # set the previous num_adver workers as adversaries

            origin, target = 1, 7 # the original label and the targeted label


            adver_feeders = label_flipping(adver_feeders, origin, target)
            for adver_idx in range(num_adver):
                model.set_params(round_model)  # 'download' global model

                # perform local training
                adver_feeder = adver_feeders[adver_idx]
                adver_feeder.activate()
                for k in range(K):
                    x, y = adver_feeder.next_batch(B)
                    _, _ = model.train_step(x, y)
                adver_feeder.deactivate()

                # 'upload' model deltas
                client_model = model.get_params_numpy()
                client_grad = round_model - client_model
                if compress == True:
                    if adver_idx in errs.keys():
                        client_grad += errs[adver_idx]  # error feedback
                    reducer = Reducer(rank, reuse=True)
                    comp_grad, err = reducer.reduce(client_grad.params)  # 'upload' compress gradients
                    errs[adver_idx] = err  # local error

                    # server decompress gradients
                    decomp_grad = []
                    for i in range(len(comp_grad)):
                        if type(comp_grad[i]) == list:  # grads in layer i that being compressed
                            p = comp_grad[i][0]
                            q = comp_grad[i][1]
                            decomp_grad.append(reducer.decompress(p, q, round_model.params[i]))
                        else:
                            decomp_grad.append(comp_grad[i])
                    client_grad = NumpyModel(decomp_grad)

                client_grads.append(client_grad)

        elif attack == 'arbitrary':
            adver_feeders = data_feeders[:num_adver]  # set the previous num_adver workers as adversaries
            for adver_idx in range(num_adver):
                model.set_params(round_model)  # 'download' global model

                # perform local training
                adver_feeder = adver_feeders[adver_idx]
                adver_feeder.activate()
                for k in range(K):
                    x, y = adver_feeder.next_batch(B)
                    _, _ = model.train_step(x, y)
                adver_feeder.deactivate()

                # 'upload' model deltas
                client_model = model.get_params_numpy()
                client_grad = client_model - round_model # the negative true gradients

                if compress == True:
                    if adver_idx in errs.keys():
                        client_grad += errs[adver_idx]  # error feedback
                    reducer = Reducer(rank, reuse=True)
                    comp_grad, err = reducer.reduce(client_grad.params)  # 'upload' compress gradients
                    errs[adver_idx] = err  # local error

                    # server decompress gradients
                    decomp_grad = []
                    for i in range(len(comp_grad)):
                        if type(comp_grad[i]) == list:  # grads in layer i that being compressed
                            p = comp_grad[i][0]
                            q = comp_grad[i][1]
                            decomp_grad.append(reducer.decompress(p, q, round_model.params[i]))
                        else:
                            decomp_grad.append(comp_grad[i])
                    client_grad = NumpyModel(decomp_grad)

                client_grads.append(client_grad)

        if detection == True:
            detect = Detect(client_grads, round_model)
            select_clients = detect.detect(unlabeled_data, model)
            round_agg = aggregate(select_clients, round_agg)
        else:
            round_agg = aggregate(client_grads, round_agg)
        # 'pseudogradient' is the average of client model deltas
        grads = round_agg / (num_adver + num_honest)


        # produce new global model
        if compress == True:
            # apply Nesterov's momentum
            m_prev = m
            m = beta * m - lr * grads
            grads = beta * m_prev - (1 + beta) * m

        round_model = round_model - grads

        if t % test_freq == 0:
            model.set_params(round_model)
            test_errs[t], test_accs[t] = model.test(test_data[0],
                                                            test_data[1],
                                                            test_B)

    return test_errs, test_accs


def run_fedavg(data_feeders, test_data, model,
               T, C, K, B, rank, seed, test_freq=1, test_B=64, detection=False, compress=False):
    """
    Args:
    - data_feeders: {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - T:            {int} number of rounds of FL
    - C:            {float} the fraction of participated clients in each round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - slr:          {float} if the compression mode is turned on, this is the learning rate that performed by the server
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    - detection:    {bool} the mode of detection
    - compress:     {bool} the mode of compression

    Returns: test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then
    the test arrays will have 0s in the non-tested rounds.
    """
    np.random.seed(seed)

    test_errs = np.zeros((T), dtype=np.float32)
    test_accs = np.zeros((T), dtype=np.float32)

    unlabeled_size = 100
    unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]

    round_model = model.get_params_numpy()  # current global model
    round_agg = model.get_params_numpy()  # client aggregate model

    num_usr = data_feeders.__len__()  # the number of total users
    M = int(num_usr * C)  # the number of workers participated in each round

    if compress == True:
        errs = {}
        # momentum setting
        lr = 1
        beta = 0.9
        m = round_model.zeros_like()

    if detection == True:
        MIs = {}

    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        # id's of workers participating in this round
        user_idxs = np.random.choice(num_usr, M, replace=False)

        client_grads = []

        for user_idx in user_idxs:
            model.set_params(round_model)  # 'download' global model

            # perform local training
            data_feeders[user_idx].activate()
            for k in range(K):
                x, y = data_feeders[user_idx].next_batch(B)
                _, _ = model.train_step(x, y)
            data_feeders[user_idx].deactivate()

            # get model deltas
            client_model = model.get_params_numpy()
            client_grad = round_model - client_model
            if compress == True:
                if user_idx in errs.keys():
                    client_grad += errs[user_idx]  # error feedback
                reducer = Reducer(rank, reuse=True)
                comp_grad, err = reducer.reduce(client_grad.params) # 'upload' compress gradients
                errs[user_idx] = err  # local error

                # server decompress gradients
                decomp_grad = []
                for i in range(len(comp_grad)):
                    if type(comp_grad[i]) == list: # grads in layer i that being compressed
                        p = comp_grad[i][0]
                        q = comp_grad[i][1]
                        decomp_grad.append(reducer.decompress(p, q, round_model.params[i]))
                    else:
                        decomp_grad.append(comp_grad[i])
                client_grad = NumpyModel(decomp_grad)

            client_grads.append(client_grad)

        if detection == True:
            detect = Detect(client_grads, round_model)
            select_clients = detect.detect(unlabeled_data, model)
            round_agg = aggregate(select_clients, round_agg)
        else:
            round_agg = aggregate(client_grads, round_agg)
            # 'pseudogradient' is the average of client model deltas

        grads = round_agg / M

        if compress == True:
            # apply Nesterov's momentum
            m_prev = m
            m = beta * m -lr * grads
            grads = beta * m_prev - (1 + beta) * m

        round_model = round_model - grads

        if t % test_freq == 0:
            model.set_params(round_model)
            test_errs[t], test_accs[t] = model.test(test_data[0],
                                                    test_data[1],
                                                    test_B)

    return test_errs, test_accs
