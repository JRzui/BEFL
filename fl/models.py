"""
Pytorch models for use with the FEMNIST, CIFAR100, Shakespeare, StackOverflow FL
simulations. Also contains the NumpyModel class for conveniently containing and 
performing operations on an entire model/set of values at once.
"""
import torch
import numpy as np 
import numbers
import operator


class FLModel(torch.nn.Module):
    """
    Extension of the pytorch Module class that provides methods for easily 
    extracting/loading model params, training, calculating gradients etc. when
    writing FL loops.
    """
    
    def __init__(self, device):
        """
        Return a new FL model with model layers placed on device.
        
        Args:
        - device:   {torch.device} where to place model
        """
        super(FLModel, self).__init__()
        self.optim      = None
        self.device     = device
        self.loss_fn    = None

    def set_optim(self, optim):
        """
        Allocates an optimizer for this model to use during training.
        
        Args:
        - optim:    {torch.optim.optimizer}
        """
        self.optim = optim

    def get_params(self):
        """
        Returns copies of model parameters as a list of Numpy ndarrays.
        """
        return [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]
        
    def get_params_numpy(self):
        """
        Returns copy of model parameters as a NumpyModel.
        """
        return NumpyModel(self.get_params())
        
    def set_params(self, new_params):
        """
        Set all the parameters of this model (values are copied).
        
        Args:
        - new_params: {list, NumpyModel} all ndarrays must be same shape as 
                      model params
        """
        with torch.no_grad():
            for (p, new_p) in zip(self.parameters(), new_params):
                p.copy_(torch.tensor(new_p))
   
    def forward(self, x):
        """
        Return the result of a forward pass of this model. 
        
        Args:
        - x:    {torch.tensor} with shape: [batch_size, sample_shape]
        
        Returns:
        {torch.tensor} with shape: [batch_size, output_shape]
        """
        raise NotImplementedError()
        
    def calc_acc(self, x, y):
        """
        Return the performance metric (not necessarily accuracy) of the model 
        with inputs x and target y.
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
        {float} mean performance metric across batch
        """
        raise NotImplementedError()
    
    def train_step(self, x, y):
        """
        Perform a single step of training using samples x and targets y. The 
        set_optim method must have been called with a torch.optim.optimizer 
        before using this method.
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
        (float, float) loss and performance metric for given x, y
        """
        logits  = self.forward(x)           # forward pass            
        loss    = self.loss_fn(logits, y)
        acc     = self.calc_acc(logits, y)
        self.optim.zero_grad()              # reset model gradient tensors
        loss.backward()
        self.optim.step()
        
        return loss.item(), acc

    def calc_grads_numpy(self, feeder, B):
        """
        Return the average gradients over all samples contained in feeder as a
        NumpModel.
        
        Args:
        - feeder:   {PyTorchDataFeeder} containing samples and labels
        - B:        {int} batch size to use while calculating grads
        
        Returns:
        {NumpyModel} containing average gradients
        """
        n_batches = int(np.ceil(feeder.n_samples / B))
        grads = None
        
        for b in range(n_batches):
            x, y    = feeder.next_batch(B)
            err     = self.loss_fn(self.forward(x), y)
            self.optim.zero_grad()      # reset model gradient tensors
            err.backward()              # gradients calculated here
            
            # get all batch gradients as a NumpyModel
            batch_grads = NumpyModel([np.copy(p.grad.cpu().numpy()) 
                                        for p in self.parameters()])
            
            if grads is None:
                grads = batch_grads
            else:
                grads = grads + batch_grads
        
        return grads / n_batches
        
    def test(self, x, y, B):
        """
        Return the average error and performance metric over all samples.
        
        Args:
        - x: {torch.tensor} of shape [num_samples, input_shape]
        - y: {torch.tensor} of shape [num_samples, output_shape]
        - B: {int} batch size to use whilst testing
        
        Returns:
        
        """
        n_batches   = int(np.ceil(x.shape[0] / B))
        err         = 0.0       # cumulative error
        acc         = 0.0       # cumulative performance metric
        
        for b in range(n_batches):
            logits  = self.forward(x[b*B:(b+1)*B])
            err     += self.loss_fn(logits, y[b*B:(b+1)*B]).item()
            acc     += self.calc_acc(logits, y[b*B:(b+1)*B])
            
        return err/n_batches, acc/n_batches

class FEMNISTModel(FLModel):
    """
    A Convolutional (conv) model for use with the FEMNIST dataset, using 
    standard cross entropy loss. Model layers consist of:
    - 3x3 conv, stride 1, 32 filters, ReLU
    - 2x2 max pooling, stride 2
    - 3x3 conv, stride 1, 64 filters, ReLU
    - 2x2 max pooling, stride 2
    - 512 neuron fully connected, ReLU
    - 62 neuron softmax output
    """
    
    def __init__(self, device):
        """
        Return a new FEMNISTModel, parameters stored on device.
        
        Args:
        - device:   {torch.device} where to place model
        """
        super(FEMNISTModel, self).__init__(device)
        self.loss_fn    = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.conv1      = torch.nn.Conv2d(1, 32, 3, 1).to(device)
        self.relu1      = torch.nn.ReLU().to(device)
        self.pool1      = torch.nn.MaxPool2d(2, 2).to(device)
        
        self.conv2      = torch.nn.Conv2d(32, 64, 3, 1).to(device)
        self.relu2      = torch.nn.ReLU().to(device)
        self.pool2      = torch.nn.MaxPool2d(2, 2).to(device)
        
        self.flat       = torch.nn.Flatten().to(device)
        self.fc1        = torch.nn.Linear(1600, 512).to(device)
        self.relu3      = torch.nn.ReLU().to(device)
        
        self.out        = torch.nn.Linear(512, 62).to(device)
        
    def forward(self, x):
        a = self.pool1(self.relu1(self.conv1(x)))
        b = self.pool2(self.relu2(self.conv2(a)))
        c = self.relu3(self.fc1(self.flat(b)))
        
        return self.out(c)
        
    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()

class CIFAR10Model(FLModel):
    """
    Return a new CIFAR10 model, using the ResNet model from paper
    He et.al. "Deep Residual Learning for Image Recognition", using
    standard cross entropy loss.

    Args:
        - device:   {torch.device} where to place model
    """

    def __init__(self, device):
        super(CIFAR10Model, self).__init__(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.in_channels = 16
        self.conv = conv3x3(3, 16).to(device)
        self.bn = torch.nn.BatchNorm2d(16).to(device)
        self.relu = torch.nn.ReLU(inplace=True).to(device)
        self.layer1 = self.make_layer(ResidualBlock, 16, 2).to(device)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, 2).to(device)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, 2).to(device)
        self.avg_pool = torch.nn.AvgPool2d(8).to(device)
        self.fc = torch.nn.Linear(64, 10).to(device)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()

def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class NumpyModel():
    """
    A convenient class for containing an entire model/set of optimiser values. 
    Operations (+, -, *, /, **) can then be done on a whole model/set of values 
    conveniently.
    """
    
    def __init__(self, params):
        """
        Returns a new NumpyModel.
        
        Args:
        - params:  {list} of Numpy ndarrays/pytorch tensors 
        """
        self.params = params
        
    def _op(self, other, f):
        """
        Check type of other and perform function f on values contained in this
        NumpyModel.
        
        Args:
        - other:    {int, float, NumpyArray}
        - f:        number-returning function to apply
        
        Returns:
        The NumpyModel produced as a result of applying f to self and other.
        """
        if isinstance(other, numbers.Number):
            new_params = [f(p, other) for p in self.params]
            
        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]
            
        else:
            raise ValueError('Incompatible type for op: {}'.format(other))
        
        return NumpyModel(new_params)
        
    def __array_ufunc__(self, *args, **kwargs):
        """
        If an operation between a Numpy scalar/array and a NumpyModel has the 
        numpy value first (e.g. np.float32 * NumpyModel), Numpy will attempt to 
        broadcast the value to the NumpyModel, which acts as an iterable. This 
        results in a NumpyModel *not* being returned from the operation. The 
        explicit exception prevents this from happening silently. To fix, put 
        the NumpyModel first in the operation, e.g. (NumpyModel * np.float32) 
        instead of (np.float32 * NumpyModel), which will call the NumpModel's 
        __mul__, instead of np.float32's.
        """
        raise NotImplementedError(  "Numpy attempted to broadcast to a "
                                  + "NumpyModel. See docstring of "
                                  + "NumpyModel's __array_ufunc__")
        
        
    def copy(self):
        """
        Return a new NumpyModel with copied values.
        """
        return NumpyModel([np.copy(p) for p in self.params])
        
    def abs(self):
        """
        Return a new NumpyModel with all absolute values.
        """
        return NumpyModel([np.abs(p) for p in self.params])
        
    def zeros_like(self):
        """
        Return a new NumpyModel with same shape, but with 0-filled params.
        """
        return NumpyModel([np.zeros_like(p) for p in self.params])
        
    def __add__(self, other):
        """
        Return the NumpyModel resulting from the addition of self and other.
        """
        return self._op(other, operator.add)
        
    def __radd__(self, other):
        """
        Return the NumpyModel resulting from the addition of other and self.
        """
        return self._op(other, operator.add)

    def __sub__(self, other):
        """
        Return the NumpyModel resulting from the subtraction of other from self.
        """
        return self._op(other, operator.sub)
        
    def __mul__(self, other):
        """
        Return the NumpyModel resulting from the multiply of self and other.
        """
        return self._op(other, operator.mul)
        
    def __rmul__(self, other):
        """
        Return the NumpyModel resulting from the multiply of other and self.
        """
        return self._op(other, operator.mul)
        
    def __truediv__(self, other):
        """
        Return the NumpyModel resulting from the division of self by other.
        """
        return self._op(other, operator.truediv)
        
    def __pow__(self, other):
        """
        Return the NumpyModel resulting from taking self to the power of other.
        """
        return self._op(other, operator.pow)
        
    def __getitem__(self, key):
        """
        Get param at index key.
        
        Args:
        - key:  int, index of parameter to retrieve
        
        Returns:
        Numpy ndarray param at index key
        """
        return self.params[key]
        
    def __len__(self):
        """
        Returns number of params (Numpy ndarrays) contained in self.
        """
        return len(self.params)
        
    def __iter__(self):
        """
        Returns an iterator over the parameters contained in this NumpyModel.
        """
        for p in self.params:
            yield p
