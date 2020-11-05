import torch
import types
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from timm.models.layers import create_conv2d, create_pool2d
import timm
from torch import nn
import numpy as np
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SRNet_layer1(nn.Module):

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        self.conv = create_conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, dilation=1, padding='')
        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SRNet_layer2(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer

        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)

        self.conv = create_conv2d(self.out_channels, self.out_channels,
                                  kernel_size=3, stride=1, dilation=1, padding='')

        self.norm = norm_layer(self.out_channels, **norm_kwargs)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = torch.add(x,inputs)
        return x
    
    
class SRNet_layer3(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer3, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        
        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
        self.conv = create_conv2d(self.out_channels, self.out_channels, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.pool = create_pool2d(pool_type='avg', kernel_size=3, stride=2, padding='')
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
        self.resconv = create_conv2d(self.in_channels, self.out_channels, 
                                  kernel_size=1, stride=2, dilation=1, padding='')
        
        self.resnorm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        res = self.resconv(inputs)
        res = self.resnorm(res)
        x = torch.add(res,x)
        return x
    
    
class SRNet_layer4(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet_layer4, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm_layer = norm_layer
        
        self.layer1 = SRNet_layer1(self.in_channels, self.out_channels, self.activation,
                                  norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
        self.conv = create_conv2d(self.out_channels, self.out_channels, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.norm = norm_layer(self.out_channels, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.conv(x)
        x = self.norm(x)
        return x
    
    
class OneHotConv_layer1(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(OneHotConv_layer1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        
        self.conv_dilated = create_conv2d(self.in_channels, self.out_channels//2, 
                                  kernel_size=3, stride=1, dilation=8, padding='')
        
        self.norm_dilated = norm_layer(self.out_channels//2, **norm_kwargs)
        
        self.conv = create_conv2d(self.in_channels, self.out_channels//2, 
                                  kernel_size=3, stride=1, dilation=1, padding='')
        
        self.norm = norm_layer(self.out_channels//2, **norm_kwargs)
        
    def forward(self, inputs):
        x = self.conv_dilated(inputs)
        x = self.norm_dilated(x)
        y = self.conv(inputs)
        y = self.norm(y)
        y = torch.cat((x,y), 1)
        y = self.activation(y)
        return y
    
    
class OneHotConv(nn.Module):
    def __init__(self, in_channels, nclasses, out_channels=32, global_pooling='avg', activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(OneHotConv, self).__init__()
        self.in_channels = in_channels
        self.activation = activation
        self.norm_layer = norm_layer
        self.nclasses = nclasses
        self.out_channels = out_channels
        self.global_pooling = SelectAdaptivePool2d(pool_type=global_pooling, flatten=True)
        self.fc = nn.Linear(self.out_channels, self.nclasses, bias=True)
        
        self.layer1 = OneHotConv_layer1(self.in_channels, 2*self.out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
        self.layer2 = SRNet_layer1(2*self.out_channels, self.out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs)
        
    def forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    
class SRNet(nn.Module):
    def __init__(self, in_channels, nclasses, global_pooling='avg', activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(SRNet, self).__init__()
        self.in_channels = in_channels
        self.activation = activation
        self.norm_layer = norm_layer
        self.nclasses = nclasses
        self.global_pooling = SelectAdaptivePool2d(pool_type=global_pooling, flatten=True)
        
        self.layer_1_specs = [64, 16]
        self.layer_2_specs = [16, 16, 16, 16, 16]
        self.layer_3_specs = [16, 64, 128, 256]
        self.layer_4_specs = [512]
        in_channels = self.in_channels
        
        block1 = []
        for out_channels in self.layer_1_specs:
            block1.append(SRNet_layer1(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block2 = []
        for out_channels in self.layer_2_specs:
            block2.append(SRNet_layer2(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block3 = []
        for out_channels in self.layer_3_specs:
            block3.append(SRNet_layer3(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
            
        block4 = []
        for out_channels in self.layer_4_specs:
            block4.append(SRNet_layer4(in_channels, out_channels, activation=self.activation, norm_layer=self.norm_layer, norm_kwargs=norm_kwargs))
            in_channels = out_channels
        
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        
        self.fc = nn.Linear(in_channels, self.nclasses, bias=True)
        
    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pooling(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    
class OneHotSRNet(nn.Module):
    def __init__(self, in_channels, nclasses, T, out_channels=32, global_pooling='avg', activation=nn.ReLU(), norm_layer=nn.BatchNorm2d, norm_kwargs={}):
        super(OneHotSRNet, self).__init__()
        
        self.OH = OneHotConv(in_channels*(T+1), nclasses, out_channels, global_pooling, activation, norm_layer, norm_kwargs)
        self.SRNet = SRNet(in_channels, nclasses, global_pooling, activation, norm_layer, norm_kwargs)
        
        self.merged_dim  = self.SRNet.layer_4_specs[-1]+out_channels
        self.merged_fc = nn.Linear(self.merged_dim, nclasses, bias=True)
        
    def forward(self, input_x, input_dct):
        x = self.OH.forward_features(input_dct)
        y = self.SRNet.forward_features(input_x)
        x_fc = self.OH.fc(x)
        y_fc = self.SRNet.fc(y)
        z = torch.cat((x,y),1).detach() #stop gradients
        z = self.merged_fc(z)
        return y_fc, x_fc, z