import abc
import itertools
import numpy as np

from keras.preprocessing.image import apply_affine_transform
from scipy.ndimage.interpolation import rotate as rt
import keras.backend as K


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        return

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)

        transformed_batch = x_batch.copy()   # (number, W, H, C)
        result = x_batch.copy()[:, :, :, 0]  # (number, W, H)
        result = result[:, :, :, np.newaxis] # (number, W, H, 1)
        for i, t_ind in enumerate(t_inds):
            result[i] = self._transformation_list[t_ind](transformed_batch[i])
        return result

class Transformation(object):
    def __init__(self, is_gray, k_90_rotate):
        self.gray = is_gray
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.gray:
            if x.shape[2] == 1:
                pass # input is gray
            elif x.shape[2] == 3:
                #gray = 0.07 * res_x[:, :, 2] + 0.72 * res_x[:, :, 1] + 0.21 * res_x[:, :, 0] #
                gray = ( res_x[:, :, 2] + res_x[:, :, 1] + res_x[:, :, 0] ) / 3. # original paper suggests average.
                res_x = gray[:, :, np.newaxis]
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)

        return res_x

class Transform(AbstractTransformer):
    """Transformation Set for TIAE."""
    def __init__(self):
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for is_gray, k_rotate in itertools.product((True,), range(0, 4)):
            transformation = Transformation(is_gray, k_rotate)
            transformation_list.append(transformation)

        self._transformation_list = transformation_list
