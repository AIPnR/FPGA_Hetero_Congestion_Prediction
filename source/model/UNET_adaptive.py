import torch
import torch.nn as nn
import math


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=16, image_size= 16, decoder = True):
        super().__init__()
        assert image_size in [512, 256, 128, 64, 32] , "fpga_size should be 512, 256, 128, 64, 32"

        self.layer_num =  math.ceil(math.log2(image_size))
        self.decoder = decoder

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2)) #layer_0: in_channel->feature
        
        self.down_blocks = nn.ModuleList([
            None if i == 0 else
            Block(features* 2**(i-1), features * 2**i, down=True, act="leaky", use_dropout=False) if i in [1,2,3] else
            Block(features* 2**3,     features * 2**3, down=True, act="leaky", use_dropout=False)
            for i in range(self.layer_num)  #分别记录每一层 
        ])

        self.bottleneck = nn.Sequential(nn.Conv2d(features * 2**3, features * 2**3, 4, 2, 1), nn.ReLU())

        self.bottleneck_lin = nn.Linear(features* 2**3, 1)
        
        if self.decoder:
            self.initial_up =  Block(features * 2**3, features * 2**3, down=False, act="relu", use_dropout=True)
            self.up_blocks = nn.ModuleList([
                None if i == 0 else
                Block(features * 2**i * 2, features * 2**(i-1), down=False, act="relu", use_dropout=False) if i in [1,2,3] else
                Block(features * 2**3 * 2, features * 2**3,     down=False, act="relu", use_dropout=True)
                for i in range(self.layer_num-1)
            ])
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1), 
                )

    def forward(self, x):
        downs = []
        x = self.initial_down(x); downs.append(x) #第0层
        for i in range(1, self.layer_num-1):      #第1-layer_num-2层
            x = self.down_blocks[i](x)
            downs.append(x)
        x = bottleneck = self.bottleneck(x)       #第layer_num-1层
        if not self.decoder:
            bottleneck_out = self.bottleneck_lin(bottleneck.flatten(start_dim=1))  #求每个batch的单一值输出（wirelength）
            return bottleneck_out
        else: 
            x = self.initial_up(x)
            for i in range(self.layer_num-2, 0, -1): #第layer_num-1,1层
                x = self.up_blocks[i](torch.cat([x, downs[i]], 1))
            out = self.final_up(torch.cat([x, downs[0]], 1)) #第0层
            return out

    

def test():
    feature = 64
    fpga_size = 512
    batch_num = 10
    x = torch.randn((batch_num, 3, fpga_size, fpga_size))
    model = UNET(in_channels=3, out_channels=4, features=feature, image_size = fpga_size, decoder=True)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    test()
