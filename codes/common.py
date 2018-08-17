from functools import partial

import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import configuration
from chainer.training import extensions
from chainer.datasets import get_cifar10, TransformDataset
from chainer.dataset import convert
from chainer import initializers
import numpy as np


def default_conv(in_channels, out_channels, kernel_size, nobias=False):
    return L.Convolution2D(in_channels, out_channels, kernel_size, pad=(kernel_size // 2), nobias=nobias)


class ResBlock(Chain):
    def __init__(self, Conv, n_feats, kernel_size, nobias=False, bn=False, act=F.relu, res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(Conv(n_feats, n_feats, kernel_size, nobias=nobias))
            if bn: m.append(L.BatchNormalization(n_feats))
            if i == 0: m.append(act)
        with self.init_scope():
            self.body = chainer.Sequential(*m)
        self.res_scale = res_scale

    def __call__(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res




class Upsampler(chainer.Sequential):
    """
    Upsample netowrk module using depth2space.
    """
    def __init__(self, Conv, scale, n_feats, bn=False, act=False, nobias=False):
        m = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(np.log2(scale))):
                m.append(Conv(n_feats, 4*n_feats, 3, nobias))
                m.append(partial(F.depth2space, r=2))
                if bn: m.append(L.BatchNormalization(n_feats))

                if act == 'relu':
                    m.append(F.relu)
                elif act == 'prelu':
                    m.append(L.PReLU(n_feats))
        elif scale == 3:
            m.append(Conv(n_feats, 9*n_feats, 3, nobias))
            m.append(partial(F.depth2space, r=3))
            if bn: m.append(L.BatchNormalization(n_feats))

            if act == 'relu':
                m.append(F.relu)
            elif act == 'prelu':
                m.append(L.PReLU(n_feats))

        else:
            raise NotImplementedError('scale number not implemented')

        super(Upsampler, self).__init__(*m)
