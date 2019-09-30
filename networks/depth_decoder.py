# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *



class OrdConv(nn.Module):

   def __init__(self, feats_in, num_outputs, kernel=1, padding=0, bins=80, lo=0, hi=2):
       super(OrdConv, self).__init__()

       self.conv = nn.Conv2d(feats_in, num_outputs * bins, kernel, padding=padding)
       self.num_outputs = num_outputs
       self.bins = bins

       weights = np.arange(0, bins, 1) / (bins - 1)
       weights = weights * hi + (1 - weights) * lo

       self.weights = np.zeros([1, bins, 1, 1])
       self.weights[0, :, 0, 0] = weights
       # self.weights = torch.nn.Parameter(torch.from_numpy(self.weights).type(torch.cuda.FloatTensor))
       self.weights = torch.from_numpy(self.weights).type(torch.cuda.FloatTensor)
       self.softmax = nn.Softmax(dim=1)
       self.pad = nn.ReflectionPad2d(1)
   def forward(self, x):
       x = self.pad(x)
       x = self.conv(x)

       batch_size = x.size(0)
       feat_h = x.size(2)
       feat_w = x.size(3)

       # reshape for cross entropy
       x = x.view(batch_size, self.bins, feat_h * self.num_outputs, feat_w)

       # score probabilities
       x = self.softmax(x)
       x = torch.sum((x * self.weights), dim=1)
       x = x.view(batch_size, self.num_outputs, feat_h, feat_w)
       # x = x.clamp(min=0.01)
       return x

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, use_ordConv = False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_ordConv = use_ordConv
        # decoder

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            if not self.use_ordConv:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            else:
                self.convs[("dispconv", s)] = OrdConv(self.num_ch_dec[s], self.num_output_channels, kernel=3, bins=3, lo=0, hi=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if not self.use_ordConv:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                else:
                    self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)

        return self.outputs
