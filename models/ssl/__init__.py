import torch
from torch import Tensor
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch import relu
import numpy as np

from models.ssl.vicreg import VICReg
from models.ssl.barlowtwins import BarlowTwins
from models.ssl.vibcreg import VIbCReg


def assign_ssl_method(proj_in, config, ssl_name):
    method_mapping = {
        "barlowtwins": BarlowTwins,
        "vicreg": VICReg,
        "vibcreg": VIbCReg,
    }

    assert (
        ssl_name in method_mapping
    ), f"SSL method {ssl_name} not in choices {list(method_mapping.keys())}"

    return method_mapping[ssl_name](proj_in, config)
