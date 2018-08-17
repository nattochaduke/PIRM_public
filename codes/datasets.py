import numpy as np
from chainer import dataset

from . import augmentations

import numpy as np
from chainer import dataset

from . import augmentations


class T1Dataset(dataset.DatasetMixin):
    """
    Dataset for PIRM 2018 track 1.

    argments:
        indice: a list of integers.
            A list that contains indice of the original data.
        X the name of npy file.
            The filename that has data point to be super-resolved.
        Y: the name of npy file.
            The filename that ahs data target points.
        patchsize: int
            The side of path size this class samples.
            Available only if train==True
        train: Bool
            if True, then it samples patchsize x patchsize small images.
            if False, then it samples whole image.
        spanning: Bool
            If True, then all images are arranged to be ONE image and the instance samples
            patchsize x patchsize small image from it.
            If False, the instance samples small image from each image.
            Available only if train==True
        imageconcat: float in [0, 1]
            The instance samples randomly samples and imagely concat the 2 samples
            Available only if train==True
    """

    def __init__(self, indice, X, Y, patchsize, scale=3, train=True, image_concat=0, mixup=0, flip=True, rotate=True,
                 target_bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):

        original_spec_height = 80
        original_spec_width = 160

        self.indice = indice
        self.X = np.load(X).astype(np.float32)
        self.Y = np.load(Y)[:, target_bands].astype(np.float32)
        x_bands = 14
        y_bands = len(target_bands)
        self.scale = scale
        self.patchsize = patchsize
        self.train = train
        self.image_concat = image_concat
        self.mixup = mixup
        self.rotate = rotate
        self.flip = flip

        self.sample_shape = self.X[0].shape
        if image_concat < 0 or image_concat > 1:
            raise ValueError('arg "imageconcat" must be in closed range [0, 1].')
        if mixup < 0 or mixup > 1:
            raise ValueError('arg "mixup" must be in closed range [0, 1].')
        if 0 < mixup and 0 < image_concat:
            raise ValueError('mixup and image_concat is exclusive.')

        if (not train) and image_concat > 0:
            raise ValueError('In validation or test mode, image_concat must be disabled.')

    def __len__(self):
        return len(self.indice)

    def sample(self, ind):
        uppermost = int(2*(np.random.randint(low=0, high=self.sample_shape[1] - self.patchsize)//2))
        leftmost = int(2*(np.random.randint(low=0, high=self.sample_shape[2] - self.patchsize)//2))
        # Make sure the edge coordinates are even.
        #print(uppermost, leftmost)

        x = self.X[ind][:, uppermost: uppermost + self.patchsize, leftmost: leftmost + self.patchsize].copy()
        y = self.Y[ind][:, int(self.scale * uppermost): int(self.scale * (uppermost + self.patchsize)),
            int(self.scale * leftmost): int(self.scale * (leftmost + self.patchsize))].copy()
        if self.flip:
            x, y = augmentations.random_flip(x, y)
        if self.rotate:
            x, y = augmentations.random_rotation(x, y)
        #print(x.shape, y.shape)
        return (x, y)

    def get_example(self, item):
        ind = self.indice[item]
        if self.train:
            if np.random.rand() < self.mixup:
                ind2 = np.random.choice(self.indice)
                x1, y1 = self.sample(ind)
                x2, y2 = self.sample(ind2)

                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                return (x, y)

            if np.random.rand() < self.image_concat: # conduct imageconcat or not
                ind2 = np.random.choice(self.indice)
                x1, y1 = self.sample(ind)
                x2, y2 = self.sample(ind2)
                cut_coeff = np.random.randint(low=self.patchsize//5 , high=4*self.patchsize//5)
                x = np.zeros_like(x1)
                y = np.zeros_like(y1)

                x[:, :, :cut_coeff] = x1[:, :, :cut_coeff]
                x[:, :, cut_coeff:] = x2[:, :, cut_coeff:]
                y[:, :, :int(self.scale*cut_coeff)] = y1[:, :, :int(self.scale*cut_coeff)]
                y[:, :, int(self.scale*cut_coeff):] = y2[:, :, int(self.scale*cut_coeff):]
                x, y = augmentations.random_flip(x, y)
                x, y = augmentations.random_rotation(x, y)
                return (x, y)

            else:
                x, y = self.sample(ind)
                return (x, y)

        else:
            x, y = self.X[ind], self.Y[ind]
            return (x, y)


