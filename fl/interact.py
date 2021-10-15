from numpy.core.fromnumeric import shape
import torchvision
from fl.data_utils import *
from fl.optimisers import *
from fl.detection import *

def init(name, num, P, lr):
    """
    Loading dataset and split into partitions
    Args:
    - name:     {string} the dataset name
    - num:      {int} the number of workers
    - P:        {float} the non-i.i.d extent of mnist dataset
    Returns:
    - data_feeders:     {list of PytorchDataFeeder} the datafeeders of workers
    - test_data:        {tuple of tensors} the test dataset
    - unlabeled_data:   {tensors} a fraction of test dataset without labels which is used to calculate MI
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if name == "FEMNIST":
        train_data, test_data = load_femnist('../data/FEMNIST_data/train',
                                             '../data/FEMNIST_data/test',
                                             100)  # load 1000 workers
        data_feeders = [PyTorchDataFeeder(x, torch.float32, y, 'long', device)
                        for (x, y) in zip(train_data[0], train_data[1])]
        test_data = (to_tensor(test_data[0], device, torch.float32),
                     to_tensor(test_data[1], device, 'long'))  # test data on GPU
        unlabeled_size = 100  # this is a fixed size but could be changed
        unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]

        model = FEMNISTModel(device)

    elif name == "CIFAR100":
        train, test = load_cifar100(
            '../data/CIFAR100_data/fed_cifar100_train.h5',
            '../data/CIFAR100_data/fed_cifar100_test.h5',
            num)
        crop = torchvision.transforms.RandomCrop(32, padding=4)
        flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        transform = lambda x: crop(flip(x))
        data_feeders = [PyTorchDataFeeder(x, torch.float32,
                                          y, 'long',
                                          device=torch.device('cuda:0'),
                                          transform=transform)
                        for (x, y) in zip(train[0], train[1])]
        test_data = (to_tensor(test[0], device, torch.float32),
                     to_tensor(test[1], device, 'long'))
        unlabeled_size = 100  # this is a fixed size but could be changed
        unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]

        model = CIFAR100Model(device)

    else: #MNIST dataset
        train_data, test_data = load_mnist('../data/MNIST_data', num, P)  # load the number of num workers
        data_feeders = [PyTorchDataFeeder(x, torch.float32, y, 'long', device)
                        for (x, y) in zip(train_data[0], train_data[1])]

        idxs_shuffle = np.random.permutation(num)
        data_feeders[:] = [data_feeders[i] for i in idxs_shuffle]
        test_data = (to_tensor(test_data[0], device, torch.float32),
                     to_tensor(test_data[1], device, 'long'))  # test data on GPU
        unlabeled_size = 100  # this is a fixed size but could be changed
        unlabeled_data = test_data[0][np.random.choice(test_data[0].shape[0], unlabeled_size, replace=False)]

        model = MNISTModel(device)

        shape = model_size(model)
    
    train_pre(model, lr)
    param = param_tolist(model.get_params())
    return data_feeders, test_data, unlabeled_data, model, shape, param
    
def getModel_params(model):
    """
     Args:
    - model:        {FLModel} that will perform the learning

    Returns:
    - params:       {list of list} the flattened list version of model parameters
    """
    params = model.get_params()
    params = param_tolist(params)
    return params

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

def train_pre(model, lr):
    """
    Set the optimizer and its learning rate
    Args:
    - model:        {FLModel} that will perform learning
    - lr:           {float} the learning rate
    """
    optimizer = ClientSGD(model.parameters(), lr)
    model.set_optim(optimizer)

def client_run(model, data_feeder, round_model, shape, K, B):
    """
    Args:
        - model:        {FLModel} that will perform the learning
        - user_idx:     {int} the id of the client
        - round_model:  {list of list} the global model parameters at previous round
        - K:            {int} the local training step that client performs
        - B:            {int} the batch size

    Returns:
        - client_grad   {list of list} the model updates of current training round
    """
    round_model = param_toNumpy(round_model, shape)
    model.set_params(round_model)  # 'download' global model
    # perform local training
    data_feeder.activate()
    for k in range(K):
        x, y = data_feeder.next_batch(B)
        _, _ = model.train_step(x, y)
    data_feeder.deactivate()

    # 'upload' model deltas
    client_model = model.get_params_numpy()
    client_grad = np.array(round_model) - np.array(client_model)
    return param_tolist(client_grad.tolist())

def node_agg(client_grads, round_model, unlabeled_data, model, shape):
    """
    MI based aggregatopm
    Args:
    - client_grads:     {list of list of list} the collected clients' updates
    - round_model:      {list of list} the global model params in each round
    - unlabeled_data:   {tensors} a fraction of test dataset without labels which is used to calculate MI
    - model:            {FLModel} 

    Returns:
    - round_agg:        {list of list} the aggregated updated global params
    """
    round_agg = model.get_params_numpy()
    round_agg = round_agg.zeros_like()

    #params convert
    for i in range(len(client_grads)):
        client_grads[i] = param_toNumpy(client_grads[i], shape)
    round_model = param_toNumpy(round_model, shape)

    detect = Detect(client_grads, round_model)
    select_clients = detect.detect(unlabeled_data, model)

    #FedAvg
    delta_sum = 0
    for select in select_clients:
        delta_sum += np.array(select)
    
    delta = delta_sum / len(select_clients)
    round_agg = np.array(round_model) - delta
    round_agg = param_tolist(round_agg.tolist())
    return round_agg

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

def param_tolist(params):
    """
    Args:
    - params:       {list of np.ndarray} the parameters of model updates

    Returns:
    - params_list:  {list of list} the list version of model updates
    """
    params_list = []
    for param in params:
        flat = param.reshape(-1,) #convert into a 1-D array
        params_list.append(flat.tolist())
    
    return params_list

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