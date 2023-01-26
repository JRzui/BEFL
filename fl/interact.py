import torch
from fl.data_utils import *
from fl.models import *
from fl.optimisers import ClientSGD
from fl.reducer import Reducer
from fl.aggregator import MI

def init(dsetname, lr, rank):
    """
    Loading dataset and split into partitions
    Args:
    - dsetname:     {string} the dataset name
    - lr:           {float} the learning rate of the optimizer
    Returns:
    - data_feeders:     {list of PytorchDataFeeder} the datafeeders of workers
    - test_data:        {tuple of tensors} the test dataset
    - unlabeled_data:   {tensors} a fraction of test dataset without labels which is used to calculate MI
    - model:            {FLModel} the global model
    - shape:            {list} the shape of the model parameters
    - param:            {list} the list version of the global model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dsetname == "femnist":
        # FEMNIST setting
        train_data, test_data = load_femnist('../data/FEMNIST_data/train',
                                            '../data/FEMNIST_data/test',
                                            50)  # load 50 workers
        data_feeders = [PyTorchDataFeeder(x, torch.float32, y, 'long', device)
                        for (x, y) in zip(train_data[0], train_data[1])]
        test_data = (to_tensor(test_data[0], device, torch.float32),
                    to_tensor(test_data[1], device, 'long'))  # test data on GPU

        create_model = lambda d: FEMNISTModel(d)

    elif dsetname == "cifar":
        # CIFAR10 setting
        # load 50 workers, each worker has 10 classes training data, the data volume of each class is unbalanced, the unbalance rate is set to 0.75
        train, test = load_cifar10(50, 10, 100, 0.75)
        data_feeders = [PyTorchDataFeeder(x, torch.float32,
                                              y, 'long',
                                              device=device, )
                            for (x, y) in zip(train[0], train[1])]
        test_data = (to_tensor(test[0], device, torch.float32),
                         to_tensor(test[1], device, 'long'))
        create_model = lambda d: CIFAR10Model(d)

    else:
        raise ValueError("Incorrect task name") 
        
    print("data loaded")
    model = create_model(device)
    optimizer = ClientSGD(model.parameters(), lr)
    shape = model_size(model)
    comp_shape = comp_size(model, rank)
    param = param_tolist(model.get_params())
    momentum = param_tolist(model.get_params_numpy().zeros_like())

    np.random.seed(0)
    unlabeled_size = 1000
    unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]
    
    return data_feeders, test_data, unlabeled_data, model, optimizer, shape, comp_shape, param, momentum

def attacker_run(attacker, round_model, shape, rank, K, B):
    """
    Args:
    - attacker:         {Attacker} the role of the participant
    - round_model:      {list of list} the global model parameters in each round
    - shape:            {list of tuples} the parameter shape in each layer
    - rank:             {int} the rank of the compressed matrix
    - K:                {int} the local training step
    - B:                {int} the batch size
    
    Returns:
    - {list} compressed model updates
    """
    round_model = param_toNumpy(round_model, shape)
    attacker.attack(round_model, K, B )
    grads = attacker.attack_upload(round_model, rank)
    return param_tolist(grads)
    

def honest_run(honest, round_model, shape, rank, K, B):
    """
    Args:
    - attacker:         {Worker} the role of the participant
    - round_model:      {list of list} the global model parameters in each round
    - shape:            {list of tuples} the parameter shape in each layer
    - rank:             {int} the rank of the compressed matrix
    - K:                {int} the local training step
    - B:                {int} the batch size
    
    Returns:
    - {list} compressed model updates
    """
    round_model = param_toNumpy(round_model, shape)
    honest.train_step(round_model, K, B)
    grads = honest.upload(round_model, rank)
    return param_tolist(grads)

def node_run(client_grads, round_model, momentum, beta, slr, unlabeled_data, model, shape, rank):
    """
    MI based aggregatopm
    Args:
    - client_grads:     {list of list of list} the collected clients' updates
    - round_model:      {list of list} the global model params in each round
    - beta:             {float} the momentum update hyper parameter
    - slr:              {float} the server side learning rate
    - unlabeled_data:   {tensors} a fraction of test dataset without labels which is used to calculate MI
    - model:            {FLModel} 

    Returns:
    - round_agg:        {list of list} the aggregated updated global params
    - momentum:         {list of list} the momentum value of this aggregation round
    """
    # decompress gradients
    
    params = model.get_params()
    param_dims = [param.ndim for param in params]
    
    for i in range(len(client_grads)): 
        client_grads[i] = comp_param_toNumpy(client_grads[i], shape, param_dims, rank)
    
    round_model = param_toNumpy(round_model, shape)
    momentum = param_toNumpy(momentum, shape)
    mi = MI(round_model, unlabeled_data, model, beta, slr, momentum)
    round_model, momentum = mi.apply_gradients(client_grads)
    
    round_model = param_tolist(round_model)
    momentum = param_tolist(momentum)  
    return round_model, momentum
    
def malicious_node_run(client_grads, round_model, momentum, beta, slr, unlabeled_data, model, shape, rank):
    """
    Malicious node incorrectly compute global model
    Args:
    - client_grads:     {list of list of list} the collected clients' updates
    - round_model:      {list of list} the global model params in each round
    - beta:             {float} the momentum update hyper parameter
    - slr:              {float} the server side learning rate
    - unlabeled_data:   {tensors} a fraction of test dataset without labels which is used to calculate MI
    - model:            {FLModel} 

    Returns:
    - round_agg:        {list of list} the aggregated updated global params
    - momentum:         {list of list} the momentum value of this aggregation round
    """  
    round_model = param_toNumpy(round_model, shape)
    momentum = param_toNumpy(momentum, shape)
    for i in range(len(round_model)):
        round_model[i] = np.random.normal(0, 1, round_model[i].shape)
    
    round_model = param_tolist(round_model)
    momentum = param_tolist(momentum)  
    return round_model, momentum

def test(model, shape, round_model, test_data, test_B=64):
    """
    model test
    Args:
    - model:        {FLModel} that performs the learning
    - round_model:  {list of list} the global model parameters in each round
    - test_data:    {list of tensors} test dataset
    - test_B:       {int} the test batch size, default value is 64

    Returns:
    - test_acc:     {float} the test accuracy
    """
    round_model = param_toNumpy(round_model, shape)
    model.set_params(round_model)
    _, test_acc = model.test(test_data[0],test_data[1],test_B)
    return test_acc

def comp_param_toNumpy(params, shape, ndims, rank):
    """
    Args:
    - params:       {list of list} the list version of model parameters
    - shape:        {list of tuples} the parameter shape in each layer
    - ndims:        {list of int} the param dimention of each layer

    Returns:
    - params_np:    {NumpyModel}
    """
    params_np = []
    j = 0
    reducer = Reducer(rank)
    for i in range(len(ndims)):
        if ndims[i] <= 1: # the uncompressed layer
            param_np = np.array(params[j])
            param_np = param_np.reshape(shape[i])
            params_np.append(param_np)
            j += 1
        else: # the compressed layer
            # decompressing
            p, q = np.array(params[j]), np.array(params[j+1])
            p, q = p.reshape(-1, rank), q.reshape(-1, rank)
            j += 2
            params_np.append(reducer.decompress(p, q, shape[i]))

    return NumpyModel(params_np)

def param_toNumpy(params, shape):
    """
    Args:
    - params:       {list of list} the list version of model parameters
    - shape:        {list of tuples} the parameter shape in each layer

    Returns:
    - params_np:    {list of ndarray}
    """
    params_np = []
    for i in range(len(params)):
        param_np = np.array(params[i])
        param_np = param_np.reshape(shape[i])
        params_np.append(param_np)

    return params_np

def param_tolist(params):
    """
    Args:
    - params:       {list of np.ndarray} the compressed parameters of model updates

    Returns:
    - params_list:  {list of list} the list version of model updates
    """   
    params_list = []
    for i in range(len(params)):
        if type(params[i]) == list: # grads in layer i that being compressed
            p = params[i][0] 
            p_flat = p.reshape(-1,) # convert to a 1-D array
            q = params[i][1]
            q_flat = q.reshape(-1,)
            params_list.append(p_flat)
            params_list.append(q_flat)
        else:
            flat = params[i].reshape(-1,)
            params_list.append(flat)
    return params_list

def model_size(model):
    """
    Get model size
    Args:
    - model:        {FLModel} that will perform the learning

    Returns:
    - shape:        {list of tuples} the shape of model parameters
    """
    shape = []
    if type(model) == list: #if the input is deltas
        params = model
    else:
        params = model.get_params()

    for param in params:
        shape.append(param.shape)
    return shape


def comp_size(model, rank):
    """
    Get model size
    Args:
    - model:        {FLModel} that will perform the learning

    Returns:
    - shape:        {list of int} the shape of compressed model parameters
    """
    shape = []
    if type(model) == list: #if the input is deltas
        params = model
    else:
        params = model.get_params()
    
    for param in params:
        if param.ndim <= 1: #the uncompressed layer
            shape.append(param.shape[0])
        else: #the layer being compromised
            matrix = param.reshape(param.shape[0], -1)
            n, m = matrix.shape
            rank = min(n, m , rank)
            shape.append(n*rank)
            shape.append(m*rank)
    return shape
