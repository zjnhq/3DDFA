#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch.nn as nn
from pdb import *
__all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.feature_grad = False
        self.mid_features_layer = -100
        self.low_features_layer = -100

    def forward(self, x):
        if x.type()=='torch.DoubleTensor':
            x = x.type('torch.FloatTensor')
        if x.type()== 'torch.cuda.DoubleTensor':
            x = x.type('torch.cuda.FloatTensor')
        if self.mid_features_layer<0 and self.low_features_layer<0:
            return self.forward_simple(x)
        else:
            return self.forward_store_features(x)

    def forward_simple(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    def SetFeatureLayers(self, feature_layers = [5,12]):
        if len(feature_layers)==2:
            self.low_features_layer = min(feature_layers[0], feature_layers[1])
            self.mid_features_layer = max(feature_layers[0], feature_layers[1])
        if len(feature_layers)==1:
            self.mid_features_layer = feature_layers[0]
            self.low_features_layer = - 1000
    def store_features(self, x):
        # size_d = x.size()[1] * x.size()[2] * x.size()[3]
        # if size_d
        if self.feature_layer == self.mid_features_layer:
            self.mid_features = x#.cpu().detach().numpy()
            if self.feature_grad:
                self.mid_features.retain_grad()
            else:
                self.mid_features =x.clone().detach()

        if self.feature_layer == self.low_features_layer:
            self.low_features = x#.cpu().detach().numpy()
            if self.feature_grad:
                self.low_features.retain_grad()
            else:
                self.low_features =x.clone().detach()
        # low_features_layer
    def forward_store_features(self, x):
        self.feature_layer = 0
        self.store_features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        self.feature_layer = 1
        self.store_features(x)
        x = self.relu(x)
        self.feature_layer = 2
        self.store_features(x)
        x = self.dw2_1(x)
        self.feature_layer = 3
        self.store_features(x)
        x = self.dw2_2(x)
        self.feature_layer = 4
        self.store_features(x)
        x = self.dw3_1(x)
        self.feature_layer = 5
        self.store_features(x)
        x = self.dw3_2(x)
        self.feature_layer = 6
        self.store_features(x)
        x = self.dw4_1(x)
        self.feature_layer = 7
        self.store_features(x)
        x = self.dw4_2(x)
        self.feature_layer = 8
        self.store_features(x)
        x = self.dw5_1(x)
        self.feature_layer = 9
        self.store_features(x)
        x = self.dw5_2(x)
        self.feature_layer = 10
        self.store_features(x)
        x = self.dw5_3(x)
        self.feature_layer = 11
        self.store_features(x)
        x = self.dw5_4(x)
        self.feature_layer = 12
        self.store_features(x)
        x = self.dw5_5(x)
        self.feature_layer = 13
        self.store_features(x)
        self.x = self.dw5_6(x)#.cpu().detach().numpy()
        self.feature_layer = 14
        self.store_features(x)
        x = self.dw6(self.x)
        self.feature_layer = 15
        self.store_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
    def SetMidfeatureNeedGrad(self, MidfeatureNeedGrad=False):
        self.feature_grad = MidfeatureNeedGrad


def mobilenet(widen_factor=1.0, num_classes=1000):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileNet(widen_factor=widen_factor, num_classes=num_classes)
    return model


def mobilenet_2(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_1(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_075(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_05(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_025(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model
