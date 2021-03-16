import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np


def doubleconv(in_channel, out_channel):
    dconv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, 3),
        nn.ReLU(inplace=True)
    )
    return dconv


def cropcat(input_tensor1, input_tensor2):
    x = input_tensor1.size()[2]
    y = input_tensor2.size()[2]
    print(x, y)
    delta = (y - x) // 2
    input_tensor2 = input_tensor2[:, :, delta:(y - delta), delta:(y - delta)]
    return torch.cat([input_tensor1, input_tensor2], dim=1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Encoder section
        self.dconv1 = doubleconv(1, 64)
        self.dconv2 = doubleconv(64, 128)
        self.dconv3 = doubleconv(128, 256)
        self.dconv4 = doubleconv(256, 512)
        self.dconv5 = doubleconv(512, 1024)
        # Decoder section
        self.dconv6 = doubleconv(1024, 512)
        self.dconv7 = doubleconv(512, 256)
        self.dconv8 = doubleconv(256, 128)
        self.dconv9 = doubleconv(128, 64)
        # Output section
        self.dconvf = nn.Conv2d(64, 1, 1)
        # Maxpool
        self.maxpool = nn.MaxPool2d(2, stride=2)
        # Up convolution
        self.uconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.uconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.uconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.uconv4 = nn.ConvTranspose2d(128, 64, 2, 2)

    def forward(self, x):
        # Encoder pass
        c1 = self.dconv1(x)
        p1 = self.maxpool(c1)
        c2 = self.dconv2(p1)
        p2 = self.maxpool(c2)
        c3 = self.dconv3(p2)
        p3 = self.maxpool(c3)
        c4 = self.dconv4(p3)
        p4 = self.maxpool(c4)
        c5 = self.dconv5(p4)
        print(c5.shape)
        # Decoder pass
        u1 = cropcat(self.uconv1(c5), c4)
        c6 = self.dconv6(u1)
        u2 = cropcat(self.uconv2(c6), c3)
        c7 = self.dconv7(u2)
        u3 = cropcat(self.uconv3(c7), c2)
        c8 = self.dconv8(u3)
        u4 = cropcat(self.uconv4(c8), c1)
        c9 = self.dconv9(u4)
        # Output pass
        out = self.dconvf(c9)
        print(out.shape)
        return out

if __name__ == "__main__":
    image1 = pimg.imread("/home/subhro/Pictures/index1.jpeg")
    img = torch.cuda.FloatTensor(image1)
    x = img.shape[0]
    img = img.reshape(1, 1, x, x)
    print(img.device)
    model = CNN().cuda()
    images = model(img)
    y = images.shape[2]
    output = images.reshape(y, y, 1)
    output = output.cpu()
    output = output.detach().numpy()
    plt.imshow(output)
    plt.show()



