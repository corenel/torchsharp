"""Configurations for Model."""

import argparse
import os

import torch


class BaseProfile(object):
    """Base class for all other profiles."""

    def __init__(self):
        """Init profile."""
        super(BaseProfile, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.cfg = None
        self.initialized = False

    def init(self):
        """Init arg parser."""
        # general
        self.parser.add_argument("--name", type=str, default="exp",
                                 help="name of experiment")
        self.parser.add_argument("--gpu_ids", type=int, default=[0],
                                 nargs="+",
                                 help="indices of gpu to be used")
        self.parser.add_argument("--manual-seed", type=int, default=None,
                                 help="manual random seed")
        # dataset
        self.parser.add_argument("--data-root", type=str, default="data",
                                 required=True,
                                 help="path to data root folder")
        self.parser.add_argument("--data-mean", type=float,
                                 default=[0.485, 0.456, 0.406],
                                 nargs=3, metavar=("R", "G", "B"),
                                 help="mean value of images in dataset")
        self.parser.add_argument("--data-std", type=float,
                                 default=[0.229, 0.224, 0.225],
                                 nargs=3, metavar=("R", "G", "B"),
                                 help="std value of images in dataset")
        self.parser.add_argument("--load-size", type=int,
                                 default=[256, 256],
                                 nargs=2, metavar=("height", "width"),
                                 help="image size for loading (to be cropped)")
        self.parser.add_argument("--image-size", type=int,
                                 default=[224, 224],
                                 nargs=2, metavar=("height", "width"),
                                 help="image size of network input")
        self.parser.add_argument("--batch-size", type=int, default=64,
                                 help="batch size for data loader")
        self.parser.add_argument("--num-workers", type=int, default=1,
                                 help="number of workers for data loader")
        # model
        self.parser.add_argument("--model-root", type=str, default="data",
                                 required=True,
                                 help="path to model checkpoint folder")
        self.parser.add_argument("--restore-file", type=str, default=None,
                                 help="path to model checkpoint to restore")
        self.parser.add_argument("--restore-epoch", type=str, default=None,
                                 help="epoch of model checkpoint to restore")
        self.parser.add_argument("--weight-init",
                                 default="xavier_normal",
                                 const="xavier_normal",
                                 nargs="?",
                                 choices=["normal", "uniform", "xavier_normal",
                                          "xavier_uniform", "kaiming_normal",
                                          "kaiming_uniform", "orthogonal",
                                          "sparse"],
                                 help="method for weight initialization"
                                      "(default: %(default)s)")

        # set flag
        self.initialized = True

    def parse(self):
        """Parse profile."""
        # parge args
        if not self.initialized:
            self.init()
        self.cfg = self.parser.parse_args()
        # set current gpu device
        torch.cuda.set_device(self.cfg.gpu_ids[0])

        return self.cfg

    def show(self):
        """Print current config."""
        cfg_dict = vars(self.cfg)
        print("--- current config ---")
        for k, v in cfg_dict.items():
            print("{}:{}".format(str(k), str(v)))
        print("---")

    def save(self, filepath=None):
        """Save current config."""
        # check path
        if filepath is None:
            filepath = os.path.join(self.cfg.model_root, self.cfg.name,
                                    "{}.cfg".format(self.cfg.name))
        filepath = os.path.abspath(filepath)
        if os.path.exist(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        # write config
        cfg_dict = vars(self.cfg)
        with open(filepath, "w") as f:
            for k, v in cfg_dict.items():
                f.write("{}:{}\n".format(str(k), str(v)))
        torch.save(self.cfg, "{}.pt".format(filepath))
        print("save current config to {}".format(filepath))

    def load(self, filepath=None):
        """Load saved config."""
        if filepath is None:
            filepath = os.path.join(self.cfg.model_root, self.cfg.name,
                                    "{}.cfg.pt".format(self.cfg.name))
        if os.path.exists(filepath):
            self.cfg = torch.load(filepath)
            self.initialized = True
            print("load current config from {}".format(filepath))
        else:
            raise IOError("config file not exists in {}"
                          .format(filepath))


class TrainProfile(BaseProfile):
    """Profile for training model."""

    def __init__(self):
        """Init profile."""
        super(TrainProfile, self).__init__()
        self.training = True

    def init(self):
        """Init arg parser."""
        BaseProfile.init(self)
        # general
        self.parser.add_argument("--phase", type=str,
                                 default="train", const="train", nargs="?",
                                 choices=["train", "val", "test"],
                                 help="process phase (default: %(default)s)")
        # optimizer
        self.parser.add_argument("--lr", type=float,
                                 default=1e-4,
                                 help="base learning rate for optimizer")
        self.parser.add_argument("--lr-policy",
                                 default="lambda",
                                 const="lambda",
                                 nargs="?",
                                 choices=["lambda", "step", "multistep",
                                          "exp", "plateau"],
                                 help="method for learning rate scheduler"
                                      "(default: %(default)s)")
        self.parser.add_argument("--lr_decay_factor", type=float,
                                 default=0.1,
                                 help="factor to decay every certain epochs")
        self.parser.add_argument("--lr_decay_epoch", type=int, default=[30],
                                 nargs="+",
                                 help="number of epochs to decay lr")

        # training process
        self.parser.add_argument("--num-epoch", type=int, default=200,
                                 help="number of optimizing epochs")
        self.parser.add_argument("--log-step", type=int, default=20,
                                 help="step to log history")
        self.parser.add_argument("--val-step", type=int, default=50,
                                 help="step to validate model")
        self.parser.add_argument("--save-step", type=int, default=50,
                                 help="step to save model checkpoint")


class TestProfile(BaseProfile):
    """Profile for testing."""

    def __init__(self):
        """Init profile."""
        super(TestProfile, self).__init__()
        self.training = False

    def init(self):
        """Init arg parser."""
        BaseProfile.init(self)
        # general
        self.parser.add_argument("--phase", type=str,
                                 default="train", const="train", nargs="?",
                                 choices=["train", "val", "test"],
                                 help="process phase (default: %(default)s)")
        # input
        self.parser.add_argument("-i", "--input", type=str, default=None,
                                 help="input file or directory")
        self.parser.add_argument("-o", "--output", type=str, default="output",
                                 help="output directory")
