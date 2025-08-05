import torch
import torch.nn as nn
import math

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, channels_xny=6, features=64, image_size=512):
        super().__init__()
        self.layer_num =  math.ceil(math.log2(image_size))-4

        self.initial = nn.Sequential(
            nn.Conv2d(channels_xny, features, 4, 2, 1, padding_mode="reflect",),
            nn.LeakyReLU(0.2),
        )

        self.blocks = nn.ModuleList([
            None if i == 0 else
            CNNBlock(features* 2**(i-1), features * 2**i, stride=2) if not i==self.layer_num-1 else
            CNNBlock(features* 2**(i-1), 1, stride=1)
            for i in range(self.layer_num)  #分别记录每一层 
        ])
        


    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        for i in range(1, self.layer_num):      #第1-layer_num-2层
            x = self.blocks[i](x)
        return x


def test():
    x = torch.randn((1, 3, 512, 512))
    y = torch.randn((1, 3, 512, 512))
    model = Discriminator(channels_xny=6, features=64, image_size=512)
    preds = model(x, y)
    # print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
