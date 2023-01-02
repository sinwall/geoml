import numpy as np
from scipy.spatial import distance_matrix

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


class SineFilter():
    def __init__(self, dim=2, n_filters=32, scale=None, random_state=None):
        if scale is None:
            scale = np.sqrt(dim)
        rng = np.random.default_rng(random_state)
        self.wave_numbers = 2*rng.random((dim, n_filters))-1
        self.wave_numbers *= scale * rng.random((1, n_filters)) / np.linalg.norm(self.wave_numbers, axis=1, keepdims=True)
        self.random_state = random_state

    def apply(self, pts, weights):
        return np.sum(np.sin(pts @ self.wave_numbers)*weights, axis=-2)