import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models

from .senet import se_resnext50_32x4d, senet154

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se


class SeResNext50_Unet_Loc(nn.Module):
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_Loc, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)

        # conv1_new = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # _w = encoder.layer0.conv1.state_dict()
        # _w['weight'] = torch.cat([0.5 * _w['weight'], 0.5 * _w['weight']], 1)
        # conv1_new.load_state_dict(_w)
        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4
        self.Att6 = Attention_block(1024, 256, 128)
        self.Att7 = Attention_block(512, 128, 64)
        self.Att8 = Attention_block(256, 64, 32)
        self.Att9 = Attention_block(64, 48, 24)

   
    def forward(self, x):
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        x4 = self.Att6(g=dec6,x=enc4)
        dec6 = self.conv6_2(torch.cat([dec6, x4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        x3 = self.Att7(g=dec7,x=enc3)
        dec7 = self.conv7_2(torch.cat([dec7, x3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        x2 = self.Att8(g=dec8,x=enc2)
        dec8 = self.conv8_2(torch.cat([dec8, x2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        x1 = self.Att9(g=dec9,x=enc1)
        dec9 = self.conv9_2(torch.cat([dec9, x1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeResNext50_Unet_Double(nn.Module):   
    def __init__(self, pretrained='imagenet', **kwargs):
        super(SeResNext50_Unet_Double, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.conv1 = nn.Sequential(encoder.layer0.conv1, encoder.layer0.bn1, encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(encoder.pool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4
        self.Att6 = Attention_block(1024, 256, 128)
        self.Att7 = Attention_block(512, 128, 64)
        self.Att8 = Attention_block(256, 64, 32)
        self.Att9 = Attention_block(64, 48, 24)
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        
        # representation head: 
        self.representation = nn.Conv2d(decoder_filters[-5], 256, 1, stride=1, padding=0) 
            
        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

        self._initialize_weights()

        encoder = se_resnext50_32x4d(pretrained=pretrained)
 

    def forward1(self, x):  # rep_head是一个额外的representation head
        batch_size, C, H, W = x.shape

        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        
        
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        x4 = self.Att6(g=dec6,x=enc4)
        dec6 = self.conv6_2(torch.cat([dec6, x4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        x3 = self.Att7(g=dec7,x=enc3)
        dec7 = self.conv7_2(torch.cat([dec7, x3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        x2 = self.Att8(g=dec8,x=enc2)
        dec8 = self.conv8_2(torch.cat([dec8, x2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        x1 = self.Att9(g=dec9,x=enc1)
        dec9 = self.conv9_2(torch.cat([dec9, x1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


    def forward(self, x, rep_head=True):
        dec10_0 = self.forward1(x[:, :3, :, :])
        dec10_1 = self.forward1(x[:, 3:, :, :])

        dec10 = torch.cat([dec10_0, dec10_1], 1)
        if rep_head:
            return self.res(dec10), self.representation(dec10_0 - dec10_1)

        return self.res(dec10)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
