
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, Sigmoid, Module

from ..utils import Conv3d, Conv2d


class BoundaryModule(Module):
    """
    Boundary matching module for video or audio features.
    Input:
        F_v or F_a: (B, C_f, T)
    Output:
        M_v^ or M_a^: (B, D, T)

    """

    def __init__(self, n_feature_in, n_features=(512, 128), num_samples: int = 10, temporal_dim: int = 512,
        max_duration: int = 40
    ):
        super().__init__()

        dim0, dim1 = n_features

        # (B, n_feature_in, temporal_dim) -> (B, n_feature_in, sample, max_duration, temporal_dim)
        self.bm_layer = BMLayer(temporal_dim, num_samples, max_duration)

        # (B, n_feature_in, sample, max_duration, temporal_dim) -> (B, dim0, max_duration, temporal_dim)
        self.block0 = Sequential(
            Conv3d(n_feature_in, dim0, kernel_size=(num_samples, 1, 1), stride=(num_samples, 1, 1),
                build_activation=LeakyReLU
            ),
            Rearrange("b c n d t -> b c (n d) t")
        )

        # (B, dim0, max_duration, temporal_dim) -> (B, max_duration, temporal_dim)
        self.block1 = Sequential(
            Conv2d(dim0, dim1, kernel_size=1, build_activation=LeakyReLU),
            Conv2d(dim1, dim1, kernel_size=3, padding=1, build_activation=LeakyReLU),
            Conv2d(dim1, 1, kernel_size=1, build_activation=Sigmoid),
            Rearrange("b c d t -> b (c d) t")
        )

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.bm_layer(feature)
        feature = self.block0(feature)
        feature = self.block1(feature)
        return feature


class BMLayer(Module):
    """BM Layer"""

    def __init__(self, temporal_dim: int, num_sample: int, max_duration: int, roi_expand_ratio: float = 0.5):
        super().__init__()
        self.temporal_dim = temporal_dim
        # self.feat_dim = opt['bmn_feat_dim']
        self.num_sample = num_sample
        self.duration = max_duration
        self.roi_expand_ratio = roi_expand_ratio
        self.smp_weight = self.get_pem_smp_weight()

    def get_pem_smp_weight(self):
        T = self.temporal_dim
        N = self.num_sample
        D = self.duration
        w = torch.zeros([T, N, D, T])  # T * N * D * T
        # In each temporal location i, there are D predefined proposals,
        # with length ranging between 1 and D
        # the j-th proposal is [i, i+j+1], 0<=j<D
        # however, a valid proposal should meet i+j+1 < T
        for i in range(T - 1):
            for j in range(min(T - 1 - i, D)):
                xmin = i
                xmax = (j + 1)
                # proposals[j, i, :] = [xmin, xmax]
                length = xmax - xmin
                xmin_ext = xmin - length * self.roi_expand_ratio
                xmax_ext = xmax + length * self.roi_expand_ratio
                bin_size = (xmax_ext - xmin_ext) / (N - 1)
                points = [xmin_ext + ii *
                          bin_size for ii in range(N)]
                for k, xp in enumerate(points):
                    if xp < 0 or xp > T - 1:
                        continue
                    left, right = int(np.floor(xp)), int(np.ceil(xp))
                    left_weight = 1 - (xp - left)
                    right_weight = 1 - (right - xp)
                    w[left, k, j, i] += left_weight
                    w[right, k, j, i] += right_weight
        return w.view(T, -1).float()

    def _apply(self, fn):
        self.smp_weight = fn(self.smp_weight)

    def forward(self, X):
        input_size = X.size()
        assert (input_size[-1] == self.temporal_dim)
        # assert(len(input_size) == 3 and
        X_view = X.view(-1, input_size[-1])
        # feature [bs*C, T]
        # smp_w    [T, N*D*T]
        # out      [bs*C, N*D*T] --> [bs, C, N, D, T]
        result = torch.matmul(X_view, self.smp_weight)
        return result.view(-1, input_size[1], self.num_sample, self.duration, self.temporal_dim)
