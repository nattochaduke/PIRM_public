import argparse
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import configuration
from chainer.training import extensions
from chainer.training import updaters
from chainer.datasets import get_cifar10, TransformDataset
from chainer.dataset import convert
from chainer import dataset
from chainerui.utils import save_args

from codes import datasets
from codes import edsr
from codes import augmentations
from codes import common
from codes import super_resolution
from codes import models

chainer.using_config('autotune', False)

# ws_size = 512*64*64*64
# chainer.cuda.set_max_workspace_size(ws_size)
chainer.global_config.autotune = True
chainer.global_config.type_check = False


def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="learning rate.")
    parser.add_argument('--target_zero_mean', type=str, default="False")
    parser.add_argument('--bands',type=str, default="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13")
    parser.add_argument('--patchsize', type=int, default=64)
    parser.add_argument('--image_concat', type=float, default=0, help='vertial concat augmentation happenning ratio, in [0, 1].')
    parser.add_argument('--mixup', type=float, default=0)

    parser.add_argument('--loss_coeffs', type=str, default="1, 1, 0, 0", help="(coeff for MAE, MSE, MRAE and SID.")
    parser.add_argument('--n_feats', type=int, default=256, help='the number of kernels in the convlution just after\
                                                    the concats in each dense block.')
    parser.add_argument('--n_RDBs', type=int, default=20, help='The number of RDB units.')
    parser.add_argument('--n_denselayers', type=int, default=6, help='The number of layers in each RDB.')
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--res_scale', type=float, default=1.)
    parser.add_argument('--last_relu', type=str, default="True")
    parser.add_argument('--calc_sid', type=str, default="True")
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result_t1',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--device_index', type=int, default=0)
    args = parser.parse_args()
    save_args(args, args.out)

    bands = list(map(int, args.bands.split(',')))
    loss_coeffs = list(map(float, args.loss_coeffs.split(',')))

    if args.target_zero_mean == "False":
        train_target = 'data/PIRMt1/normalized/train_hr.npy'
        val_target = 'data/PIRMt1/normalized/val_hr.npy'
    elif args.target_zero_mean == "True":
        train_target = 'data/PIRMt1/normalized/train_hr_zeromean.npy'
        val_target = 'data/PIRMt1/normalized/val_hr_zeromean.npy'
    else:
        raise ValueError("argument target_zero_mean must be 'True' or 'False'")

    if args.last_relu == "True":
        last_relu = True
    elif args.last_relu == "False":
        last_relu = False
    else:
        raise ValueError("argument last_relu must be 'True' or 'False'")

    if args.calc_sid == "True":
        calc_sid = True
    elif args.calc_sid == "False":
        calc_sid = False
    else:
        raise ValueError("argument calc_sid must be 'True' or 'False'")

    print('==========================================')
    if args.device_index >= 0:
        print('Using GPU {}'.format(args.device_index))
    else:
        print('Using CPU')
    print('target bands are: {}'.format(bands))
    print('augmentations: image_concat: {}, mixup: {}'.format(args.image_concat, args.mixup))
    print('ratios of losses mae={}, mse={}, mrae={}, sid={}'\
          .format(loss_coeffs[0], loss_coeffs[1], loss_coeffs[2], loss_coeffs[3]))
    print('Num Minibatch-size: {}'.format(args.batchsize))
    print('Num epoch: {}'.format(args.epoch))
    print('==========================================')

    device_id = args.device_index

    model = models.ResidualDenseNet(scale=3, in_channels=14, out_channels=len(bands), n_feats=args.n_feats,
                                    n_RDBs=args.n_RDBs, n_denselayers=args.n_denselayers, growth_rate=args.growth_rate,
                                    res_scale=args.res_scale)
    if len(args.resume) > 0:
        chainer.serializers.load_npz(args.resume, model)
    model = super_resolution.SuperResolution(model, device=args.device_index, loss_coeffs=loss_coeffs,
                                             target_zeromean=False, last_relu=last_relu, calc_sid=calc_sid)
    if device_id >= 0:
        model.to_gpu(device_id)

    # Create a multi node optimizer from a standard Chainer optimizer.

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    train = datasets.T1Dataset(list(range(200)),
                               'data/PIRMt1/normalized/train_lr3.npy', train_target,
                               patchsize=args.patchsize, scale=3, train=True, image_concat=args.image_concat, mixup=args.mixup,
                               target_bands=bands)
    validation = datasets.T1Dataset(list(range(20)),
                                    'data/PIRMt1/normalized/val_lr3.npy', val_target,
                                    patchsize=64, scale=3, train=False, target_bands=bands)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(validation, args.batchsize,
                                                repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=device_id)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = extensions.Evaluator(val_iter, model, device=device_id)
    trainer.extend(evaluator, trigger=(25, 'epoch'))

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ExponentialShift('alpha', 0.8),
                   trigger=training.triggers.IntervalTrigger(50, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/MAE', 'main/MSE', 'main/MRAE', 'main/loss', 'main/SID',
         'validation/main/MAE', 'validation/main/MSE', 'validation/main/MRAE', 'validation/main/SID',
         'validation/main/loss', 'elapsed_time']),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot_object(model.predictor, 't1_denseresnet_{.updater.epoch}.npz'), trigger=(50, 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, model.predictor)

    trainer.run()


if __name__ == '__main__':
    main()
