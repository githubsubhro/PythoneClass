import torch
import torch.nn as nn
import torch.nn.functional as F


def doubleconv(in_channel, out_channel):
    dconv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, 3),
        nn.ReLU(inplace=True)
    )
    return dconv

def cropcat(input_tensor1, input_tensor2 ):
    x = input_tensor1.size()[2]
    y = input_tensor2.size()[2]
    delta = (y-x)//2
    input_tensor2 = input_tensor2[:, :, delta:(y - delta), delta:(y - delta)]
    return torch.cat([input_tensor1, input_tensor2], dim=1)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.dconv1 = doubleconv(1, 64)
        self.dconv2 = doubleconv(64, 128)
        self.dconv3 = doubleconv(128, 256)
        self.dconv4 = doubleconv(256, 512)
        self.dconv5 = doubleconv(512, 1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.uconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.uconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.uconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.uconv4 = nn.ConvTranspose2d(128, 64, 2, 2)

    def forward(self, x):
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
        print(c5.device)
        u1 = self.uconv1(c5)
        u1 = cropcat(u1, c4)
        print(u1.shape)
        c6 =

if __name__ == "__main__":
    image = torch.rand(1, 1, 572, 572).cuda()
    model = CNN().cuda()
    print(image.shape)
    model(image)
