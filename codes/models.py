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

from . import common

# ================================Lines for Residual Dense Network  ======================

class make_dense(Chain):
    def __init__(self, in_feats, out_feats, kernel_size, nobias=False):
        super(make_dense, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_feats, out_feats, ksize=kernel_size, pad=(kernel_size // 2), nobias=nobias)
    def __call__(self, x):
        out = F.relu(self.conv(x))
        out = F.concat((x, out))
        return out

class RDB(Chain):
    def __init__(self, n_feats, n_denselayer, growth_rate, res_scale=1):
        super(RDB, self).__init__()
        n_channels_ = n_feats
        self.res_scale = res_scale

        m = []
        for _ in range(n_denselayer):
            m.append(make_dense(n_channels_, growth_rate, kernel_size=3, nobias=False))
            n_channels_ += growth_rate
        with self.init_scope():
            self.dense_layers = chainer.Sequential(*m)
            self.conv_1x1 = L.Convolution2D(n_channels_, n_feats, ksize=1, nobias=True) # LFF

    def __call__(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + self.res_scale * x
        return out

class Sequence_RDBs(ChainList):
    def __init__(self, n_RDBs, n_feats, n_denselayers, growth_rate, res_scale=1):
        super(Sequence_RDBs, self).__init__()
        for _ in range(n_RDBs):
            self.add_link(RDB(n_feats, n_denselayers, growth_rate, res_scale))

    def __call__(self, x):
        outs = []
        for link in self.children():
            x = link(x)
            outs.append(x)
        return outs

class ResidualDenseNet(Chain):
    """
    Network for Track 1
    https://arxiv.org/abs/1802.08797
    """
    def __init__(self, scale, in_channels, out_channels, n_feats, n_RDBs, n_denselayers, growth_rate, res_scale=1):
        """

        :param scale: int
            the upresolution scale. This must be 3 or exponential of 2.
        :param in_channels: int
            the number of channels of the input.
        :param out_channels: int
            the number of channels of the output.
        :param n_feats: int
            the number of kernels in the 1x1 conv layer just after dense concatenation.
        :param n_RDBs: int
            the number of RDB units.
        :param n_denselayers: int
            the number of layers in each RDB unit.
        :param growth_rate: int
            Let the number of feature maps of the input be G0 and this value growth rate G,
            Then in each RDB, the cth layer's input is G0 * (c-1) * G. and the output is G.
        :param res_scale:
        """
        super(ResidualDenseNet, self).__init__()
        with self.init_scope():
            self.conv1 = common.default_conv(in_channels, n_feats, kernel_size=3, nobias=False)
            self.conv2 = common.default_conv(n_feats, n_feats, kernel_size=3, nobias=False)

            self.rdbs = Sequence_RDBs(n_RDBs, n_feats, n_denselayers, growth_rate, res_scale=res_scale)
            self.conv1x1 = L.Convolution2D(None, n_feats, ksize=1, pad=0, stride=1, nobias=True)
            self.conv3 = common.default_conv(n_feats, n_feats, kernel_size=3)

            self.tail = chainer.Sequential(
                common.Upsampler(common.default_conv, scale, n_feats),
                common.default_conv(n_feats, out_channels, kernel_size=3)
            )

    def __call__(self, x):
        res1 = self.conv1(x)
        out = self.conv2(res1)
        outs = self.rdbs(out)
        out = self.conv1x1(F.concat(outs))
        out = self.conv3(out)
        out += res1
        out = self.tail(out)
        # out = F.relu(out)
        return out


# ================================Lines for Residual Dense Network (Dual Streams) ======================

class ResidualDenseNetDeconv(Chain):
    """
    Network for Track 1
    https://arxiv.org/abs/1802.08797
    """
    def __init__(self, scale, in_size, in_channels, out_channels, n_feats, n_RDBs, n_denselayers, growth_rate, res_scale=1):
        """

        :param scale: int
            the upresolution scale. This must be 3 or exponential of 2.
        :param in_size: int
            input image size.
        :param in_channels: int
            the number of channels of the input.
        :param out_channels: int
            the number of channels of the output.
        :param n_feats: int
            the number of kernels in the 1x1 conv layer just after dense concatenation.
        :param n_RDBs: int
            the number of RDB units.
        :param n_denselayers: int
            the number of layers in each RDB unit.
        :param growth_rate: int
            Let the number of feature maps of the input be G0 and this value growth rate G,
            Then in each RDB, the cth layer's input is G0 * (c-1) * G. and the output is G.
        :param res_scale:
        """
        super(ResidualDenseNetDeconv, self).__init__()
        self.scale = 1.5
        with self.init_scope():
            self.conv1 = common.default_conv(in_channels, n_feats, kernel_size=3, nobias=False)
            self.conv2 = common.default_conv(n_feats, n_feats, kernel_size=3, nobias=False)

            self.rdbs = Sequence_RDBs(n_RDBs, n_feats, n_denselayers, growth_rate, res_scale=res_scale)
            self.conv1x1 = L.Convolution2D(None, n_feats, ksize=1, pad=0, stride=1, nobias=True)
            self.conv3 = common.default_conv(n_feats, n_feats, kernel_size=3)

            self.tail_rdbs = Sequence_RDBs(4, n_feats, n_denselayers, growth_rate, res_scale=res_scale)
            self.tail_conv1x1 = L.Convolution2D(None, n_feats, ksize=1, pad=0, stride=1, nobias=True)
            self.tail_conv3 = common.default_conv(n_feats, n_feats, kernel_size=3)
            self.last = L.Convolution2D(None, 14, ksize=1, pad=0, stride=1, nobias=True)


    def __call__(self, x):
        res1 = self.conv1(x)
        out = self.conv2(res1)
        outs = self.rdbs(out)
        out = self.conv1x1(F.concat(outs))
        out = self.conv3(out)
        out += res1
        out = F.resize_images(out, (int(self.scale*out.shape[2]), int(self.scale*out.shape[3])))
        outs = self.tail_rdbs(out)
        out = self.tail_conv1x1(F.concat(outs))
        out = self.tail_conv3(F.relu(out))
        out = self.last(F.relu(out))

        # out = self.tail(out)
        # out = F.relu(out)
        return out

# ================================ Lines for EDSR =====================================


class EDSR(Chain):
    def __init__(self, in_channels, out_channels, n_resblock, n_feats, scale, res_scale, initial_channels,
                 Conv=common.default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        scale =scale
        act = F.relu
        Conv = partial(Conv)

        m_head = [Conv(in_channels, n_feats, kernel_size)]

        m_body = [
            common.ResBlock(
                Conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(Conv(n_feats, n_feats, kernel_size))

        m_tail = [
            common.Upsampler(Conv, scale, n_feats, act=False),
            common.default_conv(n_feats, out_channels,
                                kernel_size)
        ]

        with self.init_scope():
            self.head = chainer.Sequential(*m_head)
            self.body = chainer.Sequential(*m_body)
            self.tail = chainer.Sequential(*m_tail)

    def __call__(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

# ==================================Lines for U-Net =================================================

class EncoderBlock(Chain):
    def __init__(self, out_channels, n_layers):
        super(EncoderBlock, self).__init__()
        convs = []
        for _ in range(n_layers):
            convs.append(L.Convolution2D(None, out_channels, ksize=3, stride=1, pad=1))
            convs.append(F.relu)

        with self.init_scope():
            self.convs = chainer.Sequential(*convs)

    def __call__(self, x):
        x = self.convs(x)
        x = F.max_pooling_2d(x, ksize=2)
        return x


class DecoderBlock(Chain):
    def __init__(self, out_channels, n_layers):
        super(DecoderBlock, self).__init__()
        convs = []
        for _ in range(n_layers):
            convs.append(L.Convolution2D(None, 2*out_channels, ksize=3, stride=1, pad=1))
        convs.append(L.Convolution2D(2*out_channels, 4*out_channels, ksize=3, stride=1, pad=1))
        convs.append(F.relu)
        with self.init_scope():
            self.convs = chainer.Sequential(*convs)

    def __call__(self, x):
        x = self.convs(x)
        x = F.depth2space(x, 2)
        return x


class UNetEnc(chainer.ChainList):
    def __init__(self, n_layers, levels, n_kernels_base):
        super(UNetEnc, self).__init__()
        for level in range(levels):
            out_channels = n_kernels_base * 2 ** level
            conv = EncoderBlock(out_channels, n_layers)
            self.add_link(conv)
            conv.name = 'conv{}'.format(level)

    def __call__(self, x):
        features = []
        for link in self.children():
            x = link(x)
            if 'conv' in link.name:
                features.append(x)
        return features


class UNetDec(chainer.ChainList):

    def __init__(self, n_layers, levels, n_kernels_base):
        super(UNetDec, self).__init__()

        with self.init_scope():
            for level in range(levels):
                out_channels = n_kernels_base * 2 ** (levels - level - 1)
                conv = DecoderBlock(out_channels, n_layers)
                self.add_link(conv)
                conv.name = 'conv{}'.format(level)

    def __call__(self, x, features):
        for link in self.children():
            if 'conv' in link.name:
                x = F.concat((x, features.pop(-1)))
                x = link(x)
        return x


class UNet(chainer.Chain):

    def __init__(self, out_channels, n_layers, levels=3, n_kernels_base=32):
        super(UNet, self).__init__()
        bottom_channels = n_kernels_base * 2 ** (levels - 1)
        bridge = [
            L.Convolution2D(bottom_channels, bottom_channels, ksize=3, stride=1, pad=1),
            F.relu,
            L.Convolution2D(bottom_channels, bottom_channels, ksize=3, stride=1, pad=1),
            F.relu,
        ]

        tail = L.Convolution2D(None, out_channels, ksize=1)
        with self.init_scope():
            self.enc = UNetEnc(n_layers, levels, n_kernels_base)
            self.bridge = chainer.Sequential(*bridge)
            self.dec = UNetDec(n_layers, levels, n_kernels_base)
            self.tail = tail

    def __call__(self, x):
        features = self.enc(x)
        x = self.bridge(features[-1])
        x = self.dec(x, features)
        x = self.tail(x)
        return x