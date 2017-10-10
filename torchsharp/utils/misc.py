"""Helpful functions for project."""

import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def enable_cudnn_benchmark():
    """Turn on the cudnn autotuner that selects efficient algorithms."""
    if torch.cuda.is_available():
        cudnn.benchmark = True
        print("cuDNN autotuner is turned on.")


def init_random_seed(manual_seed):
    """Init random seed."""
    # generate seed
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
