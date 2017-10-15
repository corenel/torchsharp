"""High-level Model class."""

import os

import torch

from .initializer import get_initializer
from .lr_scheduler import get_scheduler
from .optimizer import get_optimizer


class BaseModel(object):
    """Base class for all other models.

    Subclass should define networks, optimizers, schedulers, criterions
    and metrics by re-implementing initialize() method.
    """

    def __init__(self):
        """Init model."""
        super(BaseModel, self).__init__()
        self.name = "BaseModel"
        self.cfg = None
        self.networks = []
        self.network_names = []
        self.optimizers = []
        self.criterions = []
        self.schedulers = []
        self.initializer = None
        self.metrics = None

    def initialize(self, cfg):
        """Init model with network and config.

        Args:
            cfg (object): Profile configuration.
        """
        self.cfg = cfg
        self.gpu_ids = cfg.gpu_ids
        self.training = cfg.training
        self.initializer = get_initializer(cfg.init_type)

    def setup_network(self, net, net_name, epoch=None, init=False):
        """Setup network and add it to model.

        Args:
            net (torch.nn.Module): Network object.
            net_name (str): Network name.
            epoch (int, optional): Epoch number to find network checkpoint.
            init (bool, optional): Whether to init weigths of network.
        """
        if epoch is not None:
            self.restore_network(net, net_name, epoch)
        if init:
            self.initializer(net)
        if torch.cuda.is_available():
            net.cuda()
        self.add_network(net, net_name)

    def setup_optimizer(self, net, optim_type=None):
        """Setup optimizer for network.

        Args:
            net (torch.nn.Module): Network object.
            optim_type (str): optimizer type.
        """
        if optim_type is None:
            optim_type = self.cfg.optimizer
        optimizer = get_optimizer(net.parameters(), optim_type)
        self.optimizers.append(optimizer)
        self.setup_scheduler(optimizer)

    def setup_optimizers(self, optim_type=None):
        """Setup optimizer for all networks in model.

        Args:
            net (torch.nn.Module): Network object.
            optim_type (str): optimizer type.
        """
        if optim_type is None:
            optim_type = self.cfg.optimizer
        for net in self.networks:
            self.setup_optimizer(net, optim_type)

    def setup_scheduler(self, optimizer):
        """Setup lr scheduler for network.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer for network.
        """
        self.schedulers.append(get_optimizer(optimizer, self.cfg))

    def add_network(self, net, net_name):
        """Add network object to model.

        Args:
            net (torch.nn.Module): Network object.
            net_name (str): Network name.
        """
        self.networks.append(net)
        self.network_names.append(net_name)
        setattr(self, net_name, net)

    def save_network(self, epoch, net, net_name):
        """Save network checkpoint.

        Args:
            epoch (int): Current epoch number.
            net (torch.nn.Module): Network object to be saved.
            net_name (str): Network name string.
        """
        savedir = os.path.join(self.cfg.model_root, self.cfg.name)
        filename = "{}-{}.pt".format(epoch, net_name)
        filepath = os.path.join(savedir, filename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        torch.save(net.state_dict(), filepath)
        print("save network {} to {}".format(net_name, filepath))

    def save_networks(self, epoch):
        """Save all networks in model.

        Args:
            epoch (int): Current epoch number.
        """
        for idx, net in enumerate(self.networks):
            self.save_network(epoch, net, self.network_names[idx])

    def restore_network(self, net, net_name, epoch=None, filepath=None):
        """Restore network checkpoint.

        Support automatically restore network in checkpoint folder
        by epoch and name, or just from manual filepath.

        Args:
            net (torch.nn.Module): Network object to be saved.
            net_name (str): Network name string.
            epoch (int, optional): Epoch number to find network checkpoint.
            filepath (str, optional): Path to network checkpoint.
        """
        # restore from default path
        if filepath is None and epoch is not None:
            filepath = os.path.join(self.cfg.model_root, self.cfg.name,
                                    "{}-{}.pt".format(epoch, net_name))
        # restore network
        if filepath is not None and os.path.exists(filepath):
            net.load_state_dict(torch.load(filepath))
            net.restored = True
            print("restore network {} from {}".format(net_name, filepath))

    def restore_networks(self, epoch):
        """Restore all networks in model.

        Args:
            epoch (int): Current epoch number.
        """
        for idx, net in enumerate(self.networks):
            self.restore_network(net, self.network_names[idx], epoch=epoch)

    def forward(self, input=None):
        """Forward network with input."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def optimize(self):
        """Optimize network."""
        raise NotImplementedError(
            "custom Model class must implement this method")

    def inference(self):
        """Inference network w/o computing gradients."""
        raise NotImplementedError(
            "custom Model class must implement this method")
