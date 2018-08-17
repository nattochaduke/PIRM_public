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
import numpy as np

from . import common

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


class PredictAndCompensateNet(Chain):
    def __init__(self, device_pred, device_comp, n_resblock, n_feats, scale, res_scale, initial_channels,
                 Conv=common.default_conv):
        super(PredictAndCompensateNet, self).__init__()
        self.device_pred = device_pred
        self.device_comp = device_comp
        self.predictor_net = EDSR(n_resblock, n_feats, scale, res_scale, initial_channels,
                                  Conv).to_gpu(device_pred)
        with self.init_scope():
            self.compensator_net = EDSR(n_resblock, n_feats, scale, res_scale, initial_channels,
                                      Conv).to_gpu(device_comp)

    def __call__(self, x_pred):
        x_comp = F.copy(x_pred, self.device_comp)
        predictions = self.predictor_net(x_pred)
        compensations = self.compensator_net(x_comp)
        out = predictions + F.copy(compensations, self.device_comp)
        return out


class SingleNetwork(Chain):
    """
    A network for superresolving color images or spectral images.
    Each mode, color or spec, is trained independently with each other and
    trained weights will be transfered to DualModel class.
    """
    def __init__(self, initial_channels, out_channels, n_feats, n_resblock_pre, n_resblock_post,
                 scale=3, res_scale=0.1, kernel_size=3, Conv=common.default_conv):
        super(SingleNetwork, self).__init__()
        act = F.relu

        head = [Conv(initial_channels, n_feats, kernel_size)]

        body_pre = [
            common.ResBlock(
                Conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_pre)
        ]

        body_post = [
            common.ResBlock(
                Conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_post)
        ]

        tail = [
            common.Upsampler(Conv, scale, n_feats, act=False),
            common.default_conv(n_feats, initial_channels, kernel_size),
        ]

        with self.init_scope():
            self.head = chainer.Sequential(*head)
            self.body_pre = chainer.Sequential(*body_pre)
            self.body_post = chainer.Sequential(*body_post)
            self.tail = chainer.Sequential(*tail)

    def __call__(self, x):
        x_res1 = self.head(x)
        x = self.body_pre(x_res1)
        x = self.body_post(x)
        x += x_res1
        x = self.tail(x)
        return x


class UBranchedNetwork(Chain):
    """
    A network for superresolving color images or spectral images.
    Each mode, color or spec, is trained independently with each other and
    trained weights will be transfered to DualModel class.
    """
    def __init__(self, mode, n_resblock_pre, n_resblock_post, color_n_feats,
                 scale=3, res_scale=0.1, kernel_size=3, # params for edsr parts
                 u_out_channels=128, u_n_layers=3, u_levels=4, u_n_kernels_base=256,
                 u_body_connected=True, u_tail_connected=True,
                 Conv=common.default_conv):
        super(UBranchedNetwork, self).__init__()
        act = F.relu
        self.u_body_connected = u_body_connected
        self.u_tail_connected = u_tail_connected

        if mode == "color":
            initial_channels = 3
            n_feats = color_n_feats
        elif mode == "spec":
            initial_channels = 14
            n_feats = 4 * color_n_feats
        else:
            raise ValueError('mode must be "color" or "spec".')

        head = [Conv(initial_channels, n_feats, kernel_size)]

        body_pre = [
            common.ResBlock(
                Conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_pre)
        ]

        u_body_adaptor = [
            L.Convolution2D(n_feats+u_out_channels, n_feats+u_out_channels, ksize=3, stride=1, pad=1),
            L.Convolution2D(n_feats+u_out_channels, n_feats+u_out_channels, ksize=3, stride=1, pad=1),
            F.relu,
            L.Convolution2D(n_feats+u_out_channels, n_feats, ksize=3, stride=1, pad=1),
            L.Convolution2D(n_feats + u_out_channels, n_feats, ksize=3, stride=1, pad=1),
            F.relu
        ]

        u_tail_adaptor = [
            L.Convolution2D(n_feats + u_out_channels, n_feats + u_out_channels, ksize=3, stride=1, pad=1),
            L.Convolution2D(n_feats + u_out_channels, n_feats + u_out_channels, ksize=3, stride=1, pad=1),
            L.Convolution2D(n_feats + u_out_channels, n_feats + u_out_channels, ksize=3, stride=1, pad=1),
            L.Convolution2D(n_feats + u_out_channels, n_feats, ksize=1, stride=1)
        ]

        body_post = [
            common.ResBlock(
                Conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_post)
        ]

        tail = [
            common.Upsampler(Conv, scale, n_feats, act=False),
            common.default_conv(n_feats, initial_channels, kernel_size),
        ]

        with self.init_scope():
            self.unet = UNet(u_out_channels, u_n_layers, u_levels, u_n_kernels_base)

            self.u_body_adaptor = chainer.Sequential(*u_body_adaptor)
            self.u_tail_adaptor = chainer.Sequential(*u_tail_adaptor)

            self.head = chainer.Sequential(*head)
            self.body_pre = chainer.Sequential(*body_pre)
            self.body_post = chainer.Sequential(*body_post)
            self.tail = chainer.Sequential(*tail)

    def connect_u_body(self):
        self.u_body_connected = True

    def connect_u_tail(self):
        self.u_tail_connected = True

    def disconnect_u_body(self):
        self.u_body_connected = False

    def disconnect_u_tail(self):
        self.u_tail_connected = False

    def __call__(self, x):
        u_out = self.unet(x)

        x_res1 = self.head(x)
        x = self.body_pre(x_res1)
        if self.u_body_connected:
            x = self.u_body_adaptor(F.concat((x, u_out)))
        x = self.body_post(x)
        x += x_res1
        if self.u_tail_connected:
            x = self.u_tail_adaptor(F.concat((x, u_out)))
        x = self.tail(x)
        return x


class DualNetwork(Chain):
    """
    A network for superresolving color and spectral images simultaneously.
    Most of the weights are inherited from traind SigleNetwork models and then
    we add bridges in order to merge spectral images' precise lower-resolution data and
    color images' higher-resolution but predicted by flownet data.
    """
    def __init__(self, n_resblock_pre, n_resblock_post, color_n_feats,
                 scale=3, res_scale=0.1, kernel_size=3, Conv=common.default_conv,
                 color_channels=3, spec_channels=14, fix_before_bridge=False):
        super(DualNetwork, self).__init__()
        act = F.relu
        spec_n_feats = 4 * color_n_feats
        self.connected = True

        # ===================Strucutes in color pathway. ==================

        color_head = [Conv(color_channels, color_n_feats, kernel_size)]

        color_body_pre = [
            common.ResBlock(
                Conv, color_n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_pre)
        ]

        color_body_post = [
            common.ResBlock(
                Conv, color_n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_post)
        ]

        color_tail = [
            common.Upsampler(Conv, scale, color_n_feats, act=False),
            common.default_conv(color_n_feats, color_channels, kernel_size),
        ]

        # ======================= Structures in spec pathway. ==================

        spec_head = [Conv(spec_channels, spec_n_feats, kernel_size)]

        spec_body_pre = [
            common.ResBlock(
                Conv, spec_n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_pre)
        ]

        spec_body_post = [
            common.ResBlock(
                Conv, spec_n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblock_post)
        ]

        spec_tail = [
            common.Upsampler(Conv, scale, spec_n_feats, act=False),
            common.default_conv(spec_n_feats, spec_channels, kernel_size),
        ]

        # ======================== bridges ====================================

        color_to_spec_bridge = [
            common.ResBlock(
                Conv, spec_n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(3)
        ]

        spec_to_color_bridge = [
            common.ResBlock(
                Conv, color_n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(3)
        ]

        # ===================================================================


        with self.init_scope():
            self.color_head = chainer.Sequential(*color_head)
            self.color_body_pre = chainer.Sequential(*color_body_pre)
            self.color_body_post = chainer.Sequential(*color_body_post)
            self.color_tail = chainer.Sequential(*color_tail)

            self.spec_head = chainer.Sequential(*spec_head)
            self.spec_body_pre = chainer.Sequential(*spec_body_pre)
            self.spec_body_post = chainer.Sequential(*spec_body_post)
            self.spec_tail = chainer.Sequential(*spec_tail)

            self.color_to_spec_bridge = chainer.Sequential(*color_to_spec_bridge)
            self.spec_to_color_bridge = chainer.Sequential(*spec_to_color_bridge)

        if self.fix_before_bridge:
            self.fix_before_bridge()


    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def fix_before_bridge(self):
        self.fix_params_before_bridges = True

    def train_before_bridge(self):
        self.fix_params_before_bridges = False

    def __call__(self, color, spec):
        color_res1 = self.color_head(color)
        color = self.color_body_pre(color_res1)

        spec_res1 = self.spec_head(spec)
        spec = self.spec_body_pre(spec_res1)

        if self.fix_params_before_bridges:
            spec.unchain_backward()
            color.unchain_backward()

        if self.connected:
            color_tmp = self.spec_to_color_bridge(F.depth2space(spec, 2))
            spec_tmp = self.color_to_spec_bridge(F.space2depth(color, 2))
            color += color_tmp
            spec += spec_tmp

        color = self.color_body_post(color)
        spec = self.spec_body_post(spec)

        color += color_res1
        spec += spec_res1

        color = self.color_tail(color)
        spec = self.spec_tail(spec)
        return color, spec

