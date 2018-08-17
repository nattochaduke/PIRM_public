from chainer import link
from chainer import reporter
from chainer import Variable
from chainer import functions as F
from chainer import Variable
import numpy as np

from . import common

eps = 1.526e-8


class SuperResolution(link.Chain):
    def __init__(self, predictor, device, loss_coeffs, last_relu=True, calc_sid=True):
        """

        :param predictor:
        :param lossfun:
        :param loss_coeffs: list of length 4
            the final loss is abs * loss_coeffs[0] + mse * loss_coeffs[1] + mrae * loss_coeffs[2] + sid * loss_coeffs[3]
        :param last_relu: Bool
            If True, the loss is calculated based on absolute value of the output of the network.
            From scratch, we first train the model using mean_absolute_error. In this phase if last_relu == True,
            then the network will learn to always output zero.
        :param calc_sid: Bool
            If True, we calculate sid(spectral information divergence). This value is unstable so when the network is
            premature sid tends to be nan.

        """
        super(SuperResolution, self).__init__()
        self.y = None
        self.loss_coeffs = loss_coeffs
        self.loss = None
        self.eps = Variable(np.array(1.526e-7).astype(np.float32))  # The contest metric uses 1e-3 as eps in uint16.
                                                                    # 1.526e-7 = 10 * (2**16 - 1) * 1e-3.
        self.eps.to_gpu(device)
        self.last_relu = last_relu
        self.calc_sid = calc_sid

        with self.init_scope():
            self.predictor = predictor

    def mrae(self, y, t):
        abs_diff = F.absolute(t - y)
        relative_abs_diff = abs_diff / (t + self.eps)
        return F.mean(relative_abs_diff)

    def sid(self, y, t):
        err = F.absolute(F.sum(y * F.log10((y + self.eps) / (t + self.eps))) + \
                         F.sum(t * F.log10((t + self.eps) / (y + self.eps))))
        return err * 0.041  # = err * 2** 16 / (14 * 240 * 480)

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            x: batch of sample data points
            t: bath of ground truth points.
        It feeds features to the predictor and compare the result with ground truth.
        Returns:
            ~chainer.Variable: Loss value.
        """

        self.y = self.predictor(*args[:-1]) # The last argument is the targets (ground truths)
        t = args[-1]

        if self.last_relu:
            self.y = F.relu(self.y)

        mae = F.mean_absolute_error(self.y, t)
        mse = F.mean_squared_error(self.y, t)
        mrae = self.mrae(self.y, t)

        self.loss = self.loss_coeffs[0] * mae + self.loss_coeffs[1] * mse \
                    + self.loss_coeffs[2] * mrae

        if self.calc_sid:
            sid = self.sid(F.relu(self.y), t)
            if self.loss_coeffs[3] != 0:
                self.loss += self.loss_coeffs[3] * sid
        else:
            sid = -1 # flag of no calculation

        reporter.report({'MSE': mse, 'MAE': mae, 'MRAE': mrae, 'SID': sid, 'loss': self.loss}, self)
        return self.loss
