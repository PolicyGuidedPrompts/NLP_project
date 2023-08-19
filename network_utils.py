import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.
    """
    layers = []
    layers.append(nn.Linear(input_size, size))
    layers.append(nn.ReLU())
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(size, output_size))
    return nn.Sequential(*layers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


def torch2np(tensor, cast_float_to_double=False):
    """
    Utility function that accepts a torch tensor and does the following:
        1. Move it to the CPU (if it's on a GPU)
        2. Convert to numpy array
        3. Optionally casts float32 to float64
    """
    tensor = tensor.cpu()
    if cast_float_to_double and tensor.dtype is torch.float32:
        tensor = tensor.double()
    return tensor.numpy()
