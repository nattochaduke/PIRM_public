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

import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import shutil
import time
import zipfile

from codes import datasets
from codes import edsr
from codes import augmentations
from codes import common
from codes import super_resolution
from codes import models
chainer.using_config('autotune', False)
device = 0

def main():
    parser = argparse.ArgumentParser(description='inference for track 2.')
    parser.add_argument('--model', type=str, help="the path to the trained model")
    parser.add_argument('--target', type=str, help="the target path to the result saved in.")
    args = parser.parse_args()

    def test_inference(snapshot, ensemble=True, datafile='data/PIRMt2/normalized/test_stack_reg.npy'):
        model = models.ResidualDenseNetDeconv(scale=1.5, in_size=64, in_channels=17,
                                              out_channels=14, n_feats=256, n_RDBs=20,
                                              n_denselayers=6,
                                              growth_rate=64, res_scale=1)
        chainer.serializers.load_npz(snapshot, model)
        model.to_gpu(0)

        def inference(data):
            preds = []
            for i in range(data.shape[0]):
                sample = Variable(np.expand_dims(data[i], 0))
                sample.to_gpu(0)
                pred = model(sample)  # (1, 14, width, height)
                pred = cuda.to_cpu(pred.data[0])  # (14, width, height)
                preds.append(pred)
            preds = np.array(preds).astype(np.float64)  # (20, 14, width, height)
            return preds

        orig_data = np.load(datafile).astype(np.float32)

        if not ensemble:
            before = time.time()
            res = inference(orig_data)
            print('it took {} seconds to infer 20 images'.format(time.time() - before))
            return res
        else:
            result = np.zeros([20, 14, 240, 480]).astype(np.float64)
            rots = [0, 1, 2, 3]
            flip = [True, False]
            for r in rots:
                for j in flip:
                    data = np.rot90(orig_data, r, axes=(2, 3))
                    if j == True:
                        data = np.flip(data, axis=3)
                    res = inference(data).astype(np.float64)

                    if j == True:
                        res = np.flip(res, axis=3)
                    res = np.rot90(res, -r, axes=(2, 3))
                    result += res

            result /= 8
        return result


    def submission(result_data, target_directory, comment="", mode="test"):
        if mode == "test":
            initial_number = 111
        elif mode == "val":
            initial_number = 101

        headerfile = '''ENVI
    Description = {{{}}}
    file type = ENVI Standard
    samples = 480
    lines = 240
    bands = 14
    interleave = BSQ
    data type = 12
    wavelength = {{
                  477.247000,
                  489.555000,
                  500.302000,
                  510.928000,
                  523.277000,
                  537.970000,
                  548.940000,
                  553.063000,
                  562.501000,
                  577.373000,
                  590.598000,
                  599.917000,
                  612.924000,
                  617.508000
    }}
    header offset = 0
    byte order = 0
    reflectance scale factor = 0.000000
    '''.format(comment)

        result = result_data * (2 ** 16 - 1)
        result = result.astype(np.uint16)
        if os.path.exists(target_directory):
            shutil.rmtree(target_directory)
        os.mkdir(target_directory)

        for i in range(20):
            result[i].tofile(target_directory + '/image_{}_tr2.fla'.format(i + initial_number))
            with open(target_directory + '/image_{}_tr2.hdr'.format(i + initial_number), 'w') as f:
                f.write(headerfile)

        files = glob.glob(target_directory + '/*')

        with zipfile.ZipFile(target_directory + '/result.zip', "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                arcname = f.split('/')[-1]
                zf.write(f, arcname=arcname)
                subprocess.call(["rm", f])


    def pipeline(snapshot, target_directory, ensemble=True, mode="test"):
        if mode == "test":
            datafile = 'data/PIRMt2/normalized/test_stack_reg.npy'

        comment = snapshot + "ensemble: " + str(ensemble)

        result = test_inference(snapshot, ensemble=ensemble, datafile=datafile)
        submission(result, target_directory, comment, mode)

    pipeline(args.model, args.target, ensemble=False, mode="test")

if __name__ == '__main__':
    main()

