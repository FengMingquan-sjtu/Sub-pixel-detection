import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(ResBlock, self).__init__()
        self.func=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size-1)//2,bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size,padding=(kernel_size-1)//2,bias=False), 
            nn.BatchNorm2d(out_channels),
        )
        if in_channels==out_channels:
            self.shortcut=None
        else:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=1,padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                )

    def forward(self,x):
        if self.shortcut==None:
            identity=x
        else:
            identity=self.shortcut(x)

        out=self.func(x)
        out+=identity
        out = F.relu(out)

        return out

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,scale_factor,kernel_size=3):
        super(UpSample, self).__init__()
        r=scale_factor
        self.up_sample = nn.Sequential(
            nn.Conv2d(in_channels,in_channels * r * r, kernel_size, padding=(kernel_size-1)//2),#keep H,W same
            nn.PixelShuffle(r),#from(*,C*r*r,H,W) to (*,C,H*r,W*r)
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),#keep H,W same
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.up_sample(x)


class SPResNet(nn.Module):
    def __init__(self, scale_factor,in_channels=3,out_channels=1,feature_size=5,num_ResBlock=1,kernel_size=3):
        super(SPResNet, self).__init__()
        #in_channels=3 default input is 3-channel color img
        #out_channels=1 default output is 1-channel gray img
        self.resBlocks1=nn.Sequential(ResBlock(in_channels,feature_size),*[ResBlock(feature_size,feature_size,kernel_size) for _ in range(num_ResBlock-1)])
        self.upsample=UpSample(feature_size,feature_size,scale_factor,kernel_size)
        self.resBlocks2=ResBlock(feature_size,out_channels)
    def forward(self,x):
        x=self.resBlocks1(x)
        x=self.upsample(x)
        x=self.resBlocks2(x)
        return x
        
