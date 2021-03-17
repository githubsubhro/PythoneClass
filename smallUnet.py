# impotrs
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np


def display(tensors):
    tensors = tensors.squeeze(0)
    tensorslice = tensors[1, :, :]
    tensor1 = tensorslice.detach()
    print(tensor1.shape)
    plt.imshow(tensor1)
    plt.show()




def dconv(inc, outc):
    convt = nn.Sequential(
        nn.Conv2d(inc, outc, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(outc, outc, 3, 1, 1)

    )
    return convt


def uconv(inc, outc):
    uconvt = nn.Sequential(
        nn.ConvTranspose2d(inc, outc, 3, 1, 1),
        nn.ReLU(inplace=True)
    )
    return uconvt


# class definition
class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        # encoder section
        self.dconv1 = dconv(1, 4)
        self.dconv2 = dconv(4, 8)
        # decoder section
        self.uconv1 = uconv(8, 4)
        self.uconv2 = uconv(4, 1)

    def forward(self, x):
        # encoder pass
        c1 = self.dconv1(x)
        display(c1)
        c2 = self.dconv2(c1)
        display(c2)
        # decoder pass
        u1 = self.uconv1(c2)
        display(u1)
        u2 = self.uconv2(u1)
        display(u2)
        return u2


if __name__ == "__main__":
    tensor = torch.rand(1, 1, 300, 300)
    print(tensor.shape)
    model = SCNN()
    output = model(tensor)
    print(output.shape)
