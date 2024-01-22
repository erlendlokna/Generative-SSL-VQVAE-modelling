from torch import nn
from torch import relu
from torch import nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, device):
        super().__init__()
        self.device = device  # Store the device

        # define layers
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = nn.BatchNorm1d(proj_hid)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = nn.BatchNorm1d(proj_hid)
        self.linear3 = nn.Linear(proj_hid, proj_out)

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to the device
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))  # Global max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)
        return out
    