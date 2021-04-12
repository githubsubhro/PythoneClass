# impotrs
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np


def display(tensors):
    u = tensors.shape[0]
    v = tensors.shape[1]
    w = tensors.shape[2]
    x = tensors.shape[3]
    tensors.squeeze(0)
    images = tensors.detach().cpu().numpy()
    images = images.reshape(w, x, v)
    print(images.shape)
    fig = plt.figure()
    for i in range(0, v):
        fig.add_subplot(1, v, i + 1)
        # print(images[:, :, i].shape)
        plt.imshow(images[:, :, i])
    plt.show()


def dconv(inc, outc):
    convt = nn.Sequential(
        nn.Conv2d(inc, outc, 3, 1, 0),
        nn.ReLU(inplace=True),
        nn.Conv2d(outc, outc, 3, 1, 0)

    )
    return convt


def uconv(inc, outc):
    uconvt = nn.Sequential(
        nn.ConvTranspose2d(inc, outc, 3, 1, 0),
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


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    image = pimg.imread('/home/subhro/Pictures/tulip.jpg')
    image = rgb2gray(image)
    plt.imshow(image)
    width = image.shape[0]
    height = image.shape[1]
    image = torch.cuda.FloatTensor(image)
    image = image.reshape(1, 1, width, height)
    print(image.shape)
    # image = torch.rand(1, 1, 638, 640).cuda()
    print(image.shape)
    model = SCNN().cuda()
    output = model(image)
