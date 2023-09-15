#**********************************************#
# The file with the neural networks blocks.    #
#                                              #
#**********************************************#

import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, last_block = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = input,
                              out_channels = output,
                              kernel_size = kernel,
                              padding = 'same',
                              padding_mode = 'circular')
        self.activation = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        return self.activation(self.conv(x))


class Conv2DResBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, dilation = 1, last_block = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = input,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.conv2 = nn.Conv2d(in_channels = output,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.activation1 = nn.ReLU(inplace = False) #False!
        self.activation2 = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        mid = self.conv1(x)
        return self.activation2(mid + self.conv2(self.activation1(mid)))


class CosMaP_FCN5(nn.Module):
    def __init__(self, out1 = 64, out2 = 256, out3 = 512, out4 = 1024, input = 41, output = 3, use_batch=True):
        super().__init__()
        self.l1 = Conv2DResBlock(input, out1)
        self.l2 = Conv2DResBlock(out1, out2)
        self.b1 = nn.BatchNorm2d(out2) if use_batch else nn.Identity()
        self.l3 = Conv2DResBlock(out2 + input, out3)
        self.l4 = Conv2DResBlock(out3 + out1, out4)
        self.b2 = nn.BatchNorm2d(out4) if use_batch else nn.Identity()
        self.l5 = Conv2DBlock(out4 + out2 + input, output, 1, last_block=True)

    def forward(self, x):
        #cosmap = x[:, 0, :, :].unsqueeze(1) #ambudant
        out1 = self.l1(x)
        out2 = self.b1(self.l2(out1))
        out3 = self.l3(torch.cat([out2, x], 1))
        out4 = self.b2(self.l4(torch.cat([out3, out1], 1)))
        return self.l5(torch.cat([out4, out2, x], 1))

    def log_weights(self, logger, i):
        logger.add_histogram('Layer1', torch.cat([torch.flatten(p) for p in self.l1.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer2', torch.cat([torch.flatten(p) for p in self.l2.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer3', torch.cat([torch.flatten(p) for p in self.l3.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer4', torch.cat([torch.flatten(p) for p in self.l4.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer5', torch.cat([torch.flatten(p) for p in self.l5.parameters() if p.requires_grad]), i)

########################################################################
#1D (jhaklzJZKJZ;jZKLJZ;jxlkzb,mnkljkln,kmn ,zjxlkazhxkj.zNKJZ.B.,zLAJSXLKSZJXNKJ..........)

class Conv1DBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, last_block = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels = input,
                              out_channels = output,
                              kernel_size = kernel,
                              padding = 'same',
                              padding_mode = 'circular')
        self.activation = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        return self.activation(self.conv(x))


class Conv1DResBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, dilation = 1, last_block = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = input,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.conv2 = nn.Conv1d(in_channels = output,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.activation1 = nn.ReLU(inplace = False) #False!
        self.activation2 = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        mid = self.conv1(x)
        return self.activation2(mid + self.conv2(self.activation1(mid)))


class CosMaP_FCN5_1D(nn.Module):
    def __init__(self, out1 = 64, out2 = 256, out3 = 512, out4 = 1024, input = 41, output = 3, use_batch=True):
        super().__init__()
        self.l1 = Conv1DResBlock(input, out1)
        self.l2 = Conv1DResBlock(out1, out2)
        self.b1 = nn.BatchNorm1d(out2) if use_batch else nn.Identity()
        self.l3 = Conv1DResBlock(out2 + input, out3)
        self.l4 = Conv1DResBlock(out3 + out1, out4)
        self.b2 = nn.BatchNorm1d(out4) if use_batch else nn.Identity()
        self.l5 = Conv1DBlock(out4 + out2 + input, output, 1, last_block=True)

    def forward(self, x):
        out1 = self.l1(x)
        out2 = self.b1(self.l2(out1))
        out3 = self.l3(torch.cat([out2, x], 1))
        out4 = self.b2(self.l4(torch.cat([out3, out1], 1)))
        return self.l5(torch.cat([out4, out2, x], 1))



####COUNT PARAMETERS####
if __name__ == '__main__':
    import numpy as np

    def count_model_parameters(model:nn.Module) -> dict:
        """Gets neural network size."""
        params_total = np.sum([p.numel() for p in model.parameters()])
        params_grad = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
        return {
            'params_total': params_total,
            'params_grad': params_grad
        }

    net = CosMaP_FCN5(out4=512, input=1, output=256)
    print(net)
    print(count_model_parameters(net))