#**********************************************#
# The file with SSIM and outer product layers. #
#                                              #
#**********************************************#

import torch
import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

##################################################

def rect_mult(mtrx, linear):
    """mtrx [btch x ch x m x n] # linear [btch x ch x m]
       -> out [btch x ch x m]."""
    return (mtrx @ torch.unsqueeze(linear, 3))[:, :, :, 0]


class A_Mult_Layer(nn.Module):
    """Turns linear tensors [btch x ch x m] and [btch x ch x n] into matrix tensor [btch x ch x m x n]"""
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.unsqueeze(x1, 3) @ torch.unsqueeze(x2, 2)

class L_Mult_Layer(nn.Module):
    """Turns linear tensor [btch x ch x n] into matrix tensor [btch x ch x n x n]"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 3) @ torch.unsqueeze(x, 2)

##################################################
from pytorch_msssim import SSIM

class SSIM_loss(SSIM):
    def __init__(self, channel=3):
        super().__init__(1.0, channel=channel, nonnegative_ssim=True)

    def forward(self, X, Y):
        return 1 - super().forward((X + 1) / 2, (Y + 1) / 2)


##################################################