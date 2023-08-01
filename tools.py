import numpy as np
from scipy.spatial import distance_matrix
from numba import jit, prange

class Embedder():
    def __init__(self, dim=2, lag=1, reduce=1, dim_raw=None, channel_last=False):
        self.dim = dim
        self.lag = lag
        self.reduce = reduce
        if dim_raw is None:
            dim_raw = dim + reduce
        self.dim_raw = dim_raw
        self.channel_last = channel_last
        A = np.stack([np.ones((dim_raw,))] + [np.linspace(0, 1, dim_raw) ** i for i in range(1, reduce)], axis=1)
        self.proj = np.linalg.svd(A)[0][:, reduce:]

    def transform(self, X):
        dim, lag, reduce, dim_raw = self.dim, self.lag, self.reduce, self.dim_raw
        if self.channel_last:
            length = X.shape[-2]
            result = np.stack([X[..., i*lag:length-(dim_raw-i-1)*lag, :] for i in range(dim_raw)], axis=-1)
        else:
            length = X.shape[-1]
            result = np.stack([X[..., i*lag:length-(dim_raw-i-1)*lag] for i in range(dim_raw)], axis=-1)
        if reduce > 0:
            result = result @ self.proj
        if self.channel_last:
            result = result.reshape(result.shape[:-2] + (-1, ))
        return result


class Weighting():
    def __init__(self, unit=1, method=None):
        if method is None:
            method = 'identity'
        self.unit = unit
        self.method = method

    def apply(self, pts):
        n_pts = pts.shape[0]
        dist_mat = distance_matrix(pts, pts)
        if self.method == 'identity':
            A = dist_mat
        elif self.method == 'exp':
            A = np.exp(-dist_mat / self.unit)
        b = np.ones(n_pts)
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        if self.method == 'identity':
            w /= (np.sum(w))**2
        return w.reshape(-1, 1)


@jit(nopython=True, parallel=True)
def _calculate_weighting_vectors(pts_ary):
    list_size = pts_ary.shape[0]
    result = np.empty(pts_ary.shape[:-1])
    n_pts = result.shape[-1]
    dim = pts_ary.shape[-1]
    for i in prange(list_size):
        pts = pts_ary[i]
        # pts_diff = np.empty((n_pts, n_pts, dim))
        # for j in prange(n_pts):
        #     pts_diff[j] = pts
        # for k in prange(n_pts):
        #     pts_diff[:, k] -= pts
        pts_diff = np.zeros((n_pts, n_pts, dim))
        for j in prange(n_pts):
            pts_diff[j] += pts
            pts_diff[:, j] -= pts
        dist_mat = np.sqrt(np.sum(np.square(pts_diff), axis=-1))
        A = np.exp(-dist_mat)
        b = np.ones(n_pts)
        w = np.linalg.lstsq(A, b, rcond=-1)[0]
        result[i] = w
    return result

def calculate_weighting_vectors(pts_lst, scale=1.):
    if isinstance(pts_lst, np.ndarray):
        shape = pts_lst.shape
        pts_lst = pts_lst.reshape((-1, ) + shape[-2:])
        result = _calculate_weighting_vectors(pts_lst/scale).reshape(shape[:-1] + (1, ))
    elif isinstance(pts_lst, list):
        result = [_calculate_weighting_vectors(pts/scale) for pts in pts_lst]
    else:
        raise ValueError
    return result


class SineFilter():
    def __init__(self, dim, n_filters, scale=1., random_state=None):
        rng = np.random.default_rng(random_state)
        self._wave_numbers = (2*rng.random((dim, n_filters))-1)
        self._wave_numbers *= (1/scale)*rng.random((1, n_filters)) / np.linalg.norm(self._wave_numbers, axis=0, keepdims=True)
        self.random_state = random_state

    def apply(self, pts, weights, batch_size=None):
        if batch_size is None:
            batch_size = pts.shape[0]
        if pts.ndim > weights.ndim:
            weights = weights[..., np.newaxis]
        result = np.empty((pts.shape[0], self._wave_numbers.shape[1]))
        for i_start in range(0, result.shape[0], batch_size):
            i_end = i_start + batch_size
            pts_batch = pts[i_start:i_end]
            weights_batch = weights[i_start:i_end]
            result[i_start:i_end] = np.sum(np.sin(pts_batch @ self._wave_numbers)*weights_batch, axis=-2)
        return result


def generate_loocv_masks(seg_indivs):
    for indiv in np.unique(seg_indivs):
        mask_train = seg_indivs != indiv
        mask_test = ~mask_train
        yield mask_train, mask_test