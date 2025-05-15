from torch import Tensor
from torch.nn import Module

from ..utils import Conv1d


class FrameLogisticRegression(Module):
    """
    Frame classifier (FC_v and FC_a) for video feature (F_v) and audio feature (F_a).
    Input:
        F_v or F_a: (B, C_f, T)
    Output:
        Y^: (B, 1, T)
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.lr_layer = Conv1d(n_features, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.lr_layer(features)
