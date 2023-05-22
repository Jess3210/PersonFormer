"""
Author: Jessica Kick
"""

import torch.nn as nn

class VGGish(nn.Module):
    def __init__(self, dicmodelpretrained, architecture='vggish'):
        super(VGGish, self).__init__()
        
        if architecture == 'vggish':
            genVggish = [64, 'M', 128, 'M', 256, 256,  'M', 512, 512, 'M']
            namePreTrained = ['features.0.weight', 'features.3.weight','features.6.weight', 'features.8.weight', 'features.11.weight', 'features.13.weight', 'fc.0.weight', 'fc.2.weight', 'fc.4.weight']
            fully_c = [4096, 4096, 128]
        
        self.fc = nn.Sequential(nn.Linear(512 * 24, fully_c[0]), nn.ReLU(inplace=True),
                                nn.Linear(fully_c[0], fully_c[1]), nn.ReLU(inplace=True),
                                nn.Linear(fully_c[1], fully_c[2]), nn.ReLU(inplace=True))
        
        self.conv_layers = self.create_conv_layers(genVggish, namePreTrained, dicmodelpretrained)
    
    def create_conv_layers(self, architecture, namearch, dicWeights):
        layers = []
        in_channels = 1
        counter = 0

        for x in architecture:
            if type(x) == int:
                out_channels = x
                cnn_arch = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                layers += [cnn_arch, nn.ReLU(inplace=True)] #nn.BatchNorm2d(x)
                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
             #for loading pretrained weights, specifically the pretrained vggish with audionet  
            #for loading pretrained weights from https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch/releases
            if dicWeights is not None:
                if counter < len(namearch):
                    weight = namearch[counter]
                    bias = namearch[counter]
                    cnn_arch.weight.data = dicWeights[weight]
                    cnn_arch.bias.data = dicWeights[bias]
                    counter += 1
        return nn.Sequential(*layers)
    
    def forward(self, audio):
        x = self.conv_layers(audio).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc[0](x)
        x = self.fc[1](x)
        x = self.fc[2](x)
        return x
        
