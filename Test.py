import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv2d(20, 20, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        print(x2.shape)
        print(x2.device)
        return x2


if __name__ == "__main__":
    image = torch.rand(1, 1, 1500, 1500).cuda()
    model = CNN().cuda()
    model(image)