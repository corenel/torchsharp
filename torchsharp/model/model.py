"""High-level Model class."""

import os

import torch


class BaseModel(object):
    """Base class for all other models."""

    def __init__(self):
        """Init model."""
        super(BaseModel, self).__init__()
        self.name = "BaseModel"
        self.cfg = None
        self.networks = []  # network name list
        self.optimizers = []
        self.lr_schedulers = []
        self.initializers = []
        self.metrics = None

    def init(self, net_dict, cfg):
        """Init model with network and config.

        Args:
            net_dict (dict): A dict of network name and object.
            cfg (object): Profile configuration.
        """
        raise NotImplementedError(
            "custom Model class must implement this method")

    def add_network(self, net, net_name):
        """Add network object to model.

        Args:
            net (torch.nn.Module): Network object.
            net_name (str): Network name.
        """
        if net_name not in self.networks:
            self.networks.append(net_name)
        setattr(self, net_name, net)

    def add_networks(self, net_dict):
        """Add networks to model.

        Args:
            net_dict (dict): A dict of network name and object.
        """
        for net_name, net in net_dict.items():
            self.add_network(net_name, net)

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

    def save_networks(self, epoch, net_names=None):
        """Save networks in model.

        Args:
            epoch (int): Current epoch number.
            net_names (list, optinal): A list of names of networks to be saved.
        """
        if net_names is None:
            net_names = self.networks
        for net_name in net_names.items():
            net = getattr(self, net_name)
            self.save_network(epoch, net, net_name)

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

    def restore_networks(self, epoch, net_names=None):
        """Restore networks in model.

        Args:
            epoch (int): Current epoch number.
            net_names (list, optinal): A list of names of nets to be loaded.
        """
        if net_names is None:
            net_names = self.networks
        for net_name in net_names:
            net = getattr(self, net_name)
            self.restore_network(net, net_name, epoch=epoch)

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
