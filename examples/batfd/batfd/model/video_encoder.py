from typing import Literal

import numpy as np
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, MaxPool3d, Module, Linear
from torchvision.models.video.mvit import MSBlockConfig, _mvit

from ..utils import Conv3d, Conv1d


class C3DVideoEncoder(Module):
    """
    Video encoder (E_v): Process video frames to extract features.
    Input:
        V: (B, C, T, H, W)
    Output:
        F_v: (B, C_f, T)
    """

    def __init__(self, n_features=(64, 96, 128, 128), v_cla_feature_in: int = 256):
        super().__init__()

        n_dim0, n_dim1, n_dim2, n_dim3 = n_features

        # (B, 3, 512, 96, 96) -> (B, 64, 512, 32, 32)
        self.block0 = Sequential(
            Conv3d(3, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim0, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 3, 3))
        )

        # (B, 64, 512, 32, 32) -> (B, 96, 512, 16, 16)
        self.block1 = Sequential(
            Conv3d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 96, 512, 16, 16) -> (B, 128, 512, 8, 8)
        self.block2 = Sequential(
            Conv3d(n_dim1, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv3d(n_dim2, n_dim2, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2))
        )

        # (B, 128, 512, 8, 8) -> (B, 128, 512, 2, 2) -> (B, 512, 512) -> (B, 256, 512)
        self.block3 = Sequential(
            Conv3d(n_dim2, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Conv3d(n_dim3, n_dim3, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool3d((1, 2, 2)),
            Rearrange("b c t h w -> b (c h w) t"),
            Conv1d(n_dim3 * 4, v_cla_feature_in, kernel_size=1, stride=1, build_activation=LeakyReLU)
        )

    def forward(self, video: Tensor) -> Tensor:
        x = self.block0(video)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MvitVideoEncoder(Module):

    def __init__(self, v_cla_feature_in: int = 256,
        temporal_size: int = 512,
        mvit_type: Literal["mvit_v2_t", "mvit_v2_s", "mvit_v2_b"] = "mvit_v2_t"
    ):
        super().__init__()
        if mvit_type == "mvit_v2_t":
            self.mvit = mvit_v2_t(v_cla_feature_in, temporal_size)
        elif mvit_type == "mvit_v2_s":
            self.mvit = mvit_v2_s(v_cla_feature_in, temporal_size)
        elif mvit_type == "mvit_v2_b":
            self.mvit = mvit_v2_b(v_cla_feature_in, temporal_size)
        else:
            raise ValueError(f"Invalid mvit_type: {mvit_type}")
        del self.mvit.head

    def forward(self, video: Tensor) -> Tensor:
        feat = self.mvit.conv_proj(video)
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.mvit.pos_encoding(feat)
        thw = (self.mvit.pos_encoding.temporal_size,) + self.mvit.pos_encoding.spatial_size
        for block in self.mvit.blocks:
            feat, thw = block(feat, thw)

        feat = self.mvit.norm(feat)
        feat = feat[:, 1:]
        feat = feat.permute(0, 2, 1)
        return feat


def generate_config(blocks, heads, channels, out_dim):
    num_heads = []
    input_channels = []
    kernel_qkv = []
    stride_q = [[1, 1, 1]] * sum(blocks)
    blocks_cum = np.cumsum(blocks)
    stride_kv = []

    for i in range(len(blocks)):
        num_heads.extend([heads[i]] * blocks[i])
        input_channels.extend([channels[i]] * blocks[i])
        kernel_qkv.extend([[3, 3, 3]] * blocks[i])

        if i != len(blocks) - 1:
            stride_q[blocks_cum[i]] = [1, 2, 2]

        stride_kv_value = 2 ** (len(blocks) - 1 - i)
        stride_kv.extend([[1, stride_kv_value, stride_kv_value]] * blocks[i])

    return {
        "num_heads": num_heads,
        "input_channels": [input_channels[0]] + input_channels[:-1],
        "output_channels": input_channels[:-1] + [out_dim],
        "kernel_q": kernel_qkv,
        "kernel_kv": kernel_qkv,
        "stride_q": stride_q,
        "stride_kv": stride_kv
    }


def build_mvit(config, kwargs, temporal_size=512):
    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    return _mvit(
        spatial_size=(96, 96),
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=None,
        progress=False,
        patch_embed_kernel=(3, 15, 15),
        patch_embed_stride=(1, 12, 12),
        patch_embed_padding=(1, 3, 3),
        **kwargs,
    )


def mvit_v2_b(out_dim: int, temporal_size: int, **kwargs):
    config = generate_config([2, 3, 16, 3], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(config, kwargs, temporal_size=temporal_size)


def mvit_v2_s(out_dim: int, temporal_size: int, **kwargs):
    config = generate_config([1, 2, 11, 2], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(config, kwargs, temporal_size=temporal_size)


def mvit_v2_t(out_dim: int, temporal_size: int, **kwargs):
    config = generate_config([1, 2, 5, 2], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(config, kwargs, temporal_size=temporal_size)


class VideoFeatureProjection(Module):

    def __init__(self, input_feature_dim: int, v_cla_feature_in: int = 256):
        super().__init__()
        self.proj = Linear(input_feature_dim, v_cla_feature_in)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 1)


def get_video_encoder(v_cla_feature_in, temporal_size, v_encoder, ve_features):
    if v_encoder == "c3d":
        video_encoder = C3DVideoEncoder(n_features=ve_features, v_cla_feature_in=v_cla_feature_in)
    elif v_encoder == "mvit_t":
        video_encoder = MvitVideoEncoder(v_cla_feature_in=v_cla_feature_in, temporal_size=temporal_size, mvit_type="mvit_v2_t")
    elif v_encoder == "mvit_s":
        video_encoder = MvitVideoEncoder(v_cla_feature_in=v_cla_feature_in, temporal_size=temporal_size, mvit_type="mvit_v2_s")
    elif v_encoder == "mvit_b":
        video_encoder = MvitVideoEncoder(v_cla_feature_in=v_cla_feature_in, temporal_size=temporal_size, mvit_type="mvit_v2_b")
    elif v_encoder == "marlin_vit_small":
        video_encoder = VideoFeatureProjection(input_feature_dim=13824, v_cla_feature_in=v_cla_feature_in)
    elif v_encoder == "i3d":
        video_encoder = VideoFeatureProjection(input_feature_dim=2048, v_cla_feature_in=v_cla_feature_in)
    elif v_encoder == "3dmm":
        video_encoder = VideoFeatureProjection(input_feature_dim=393, v_cla_feature_in=v_cla_feature_in)
    else:
        raise ValueError(f"Invalid video encoder: {v_encoder}")
    return video_encoder
