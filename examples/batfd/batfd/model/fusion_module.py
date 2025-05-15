import torch
from torch import Tensor
from torch.nn import Sigmoid, Module

from ..utils import Conv1d


class ModalFeatureAttnBoundaryMapFusion(Module):
    """
    Fusion module for video and audio boundary maps.

    Input:
        F_v: (B, C_f, T)
        F_a: (B, C_f, T)
        M_v^: (B, D, T)
        M_a^: (B, D, T)

    Output:
        M^: (B, D, T)
    """

    def __init__(self, n_video_features: int = 257, n_audio_features: int = 257, max_duration: int = 40):
        super().__init__()

        self.a_attn_block = ModalMapAttnBlock(n_audio_features, n_video_features, max_duration)
        self.v_attn_block = ModalMapAttnBlock(n_video_features, n_audio_features, max_duration)

    def forward(self, video_feature: Tensor, audio_feature: Tensor, video_bm: Tensor, audio_bm: Tensor) -> Tensor:
        a_attn = self.a_attn_block(audio_bm, audio_feature, video_feature)
        v_attn = self.v_attn_block(video_bm, video_feature, audio_feature)

        sum_attn = a_attn + v_attn

        a_w = a_attn / sum_attn
        v_w = v_attn / sum_attn

        fusion_bm = video_bm * v_w + audio_bm * a_w
        return fusion_bm


class ModalMapAttnBlock(Module):

    def __init__(self, n_self_features: int, n_another_features: int, max_duration: int = 40):
        super().__init__()
        self.attn_from_self_features = Conv1d(n_self_features, max_duration, kernel_size=1)
        self.attn_from_another_features = Conv1d(n_another_features, max_duration, kernel_size=1)
        self.attn_from_bm = Conv1d(max_duration, max_duration, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, self_bm: Tensor, self_features: Tensor, another_features: Tensor) -> Tensor:
        w_bm = self.attn_from_bm(self_bm)
        w_self_feat = self.attn_from_self_features(self_features)
        w_another_feat = self.attn_from_another_features(another_features)
        w_stack = torch.stack((w_bm, w_self_feat, w_another_feat), dim=3)
        w = w_stack.mean(dim=3)
        return self.sigmoid(w)


class ModalFeatureAttnCfgFusion(ModalFeatureAttnBoundaryMapFusion):

    def __init__(self, n_video_features: int = 257, n_audio_features: int = 257):
        super().__init__()
        self.a_attn_block = ModalCbgAttnBlock(n_audio_features, n_video_features)
        self.v_attn_block = ModalCbgAttnBlock(n_video_features, n_audio_features)

    def forward(self, video_feature: Tensor, audio_feature: Tensor, video_cfg: Tensor, audio_cfg: Tensor) -> Tensor:
        video_cfg = video_cfg.unsqueeze(1)
        audio_cfg = audio_cfg.unsqueeze(1)
        fusion_cfg = super().forward(video_feature, audio_feature, video_cfg, audio_cfg)
        return fusion_cfg.squeeze(1)


class ModalCbgAttnBlock(ModalMapAttnBlock):

    def __init__(self, n_self_features: int, n_another_features: int):
        super().__init__(n_self_features, n_another_features, 1)