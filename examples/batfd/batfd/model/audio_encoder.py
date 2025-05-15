from typing import Literal

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Module, Sequential, LeakyReLU, MaxPool2d, Linear
from torchvision.models.vision_transformer import Encoder as ViTEncoder

from ..utils import Conv2d


class CNNAudioEncoder(Module):
    """
    Audio encoder (E_a): Process log mel spectrogram to extract features.
    Input:
        A': (B, F_m, T_a)
    Output:
        E_a: (B, C_f, T)
    """

    def __init__(self, n_features=(32, 64, 64)):
        super().__init__()

        n_dim0, n_dim1, n_dim2 = n_features

        # (B, 64, 2048) -> (B, 1, 64, 2048) -> (B, 32, 32, 1024)
        self.block0 = Sequential(
            Rearrange("b c t -> b 1 c t"),
            Conv2d(1, n_dim0, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 32, 32, 1024) -> (B, 64, 16, 512)
        self.block1 = Sequential(
            Conv2d(n_dim0, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            Conv2d(n_dim1, n_dim1, kernel_size=3, stride=1, padding=1, build_activation=LeakyReLU),
            MaxPool2d(2)
        )

        # (B, 64, 16, 512) -> (B, 64, 4, 512) -> (B, 256, 512)
        self.block2 = Sequential(
            Conv2d(n_dim1, n_dim2, kernel_size=(2, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Conv2d(n_dim2, n_dim2, kernel_size=(3, 1), stride=1, padding=(1, 0), build_activation=LeakyReLU),
            MaxPool2d((2, 1)),
            Rearrange("b f c t -> b (f c) t")
        )

    def forward(self, audio: Tensor) -> Tensor:
        x = self.block0(audio)
        x = self.block1(x)
        x = self.block2(x)
        return x


class SelfAttentionAudioEncoder(Module):

    def __init__(self, block_type: Literal["vit_t", "vit_s", "vit_b"], a_cla_feature_in: int = 256, temporal_size: int = 512):
        super().__init__()
        # The ViT configurations are from:
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        if block_type == "vit_t":
            self.n_features = 192
            self.block = ViTEncoder(
                seq_length=temporal_size,
                num_layers=12,
                num_heads=3,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_s":
            self.n_features = 384
            self.block = ViTEncoder(
                seq_length=temporal_size,
                num_layers=12,
                num_heads=6,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        elif block_type == "vit_b":
            self.n_features = 768
            self.block = ViTEncoder(
                seq_length=temporal_size,
                num_layers=12,
                num_heads=12,
                hidden_dim=self.n_features,
                mlp_dim=self.n_features * 4,
                dropout=0.,
                attention_dropout=0.
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.input_proj = Conv2d(1, self.n_features, kernel_size=(64, 4), stride=(64, 4))
        self.output_proj = Linear(self.n_features, a_cla_feature_in)

    def forward(self, audio: Tensor) -> Tensor:
        x = audio.unsqueeze(1)  # (B, 64, 2048) -> (B, 1, 64, 2048)
        x = self.input_proj(x)  # (B, 1, 64, 2048) -> (B, feat, 1, 512)
        x = rearrange(x, "b f 1 t -> b t f")  # (B, feat, 1, 512) -> (B, 512, feat)
        x = self.block(x)
        x = self.output_proj(x)  # (B, 512, feat) -> (B, 512, 256)
        x = x.permute(0, 2, 1)  # (B, 512, 256) -> (B, 256, 512)
        return x


class AudioFeatureProjection(Module):

    def __init__(self, input_feature_dim: int, a_cla_feature_in: int = 256):
        super().__init__()
        self.proj = Linear(input_feature_dim, a_cla_feature_in)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 1)


def get_audio_encoder(a_cla_feature_in, temporal_size, a_encoder, ae_features):
    if a_encoder == "cnn":
        audio_encoder = CNNAudioEncoder(n_features=ae_features)
    elif a_encoder == "vit_t":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_t", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "vit_s":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_s", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "vit_b":
        audio_encoder = SelfAttentionAudioEncoder(block_type="vit_b", a_cla_feature_in=a_cla_feature_in, temporal_size=temporal_size)
    elif a_encoder == "wav2vec2":
        audio_encoder = AudioFeatureProjection(input_feature_dim=1536, a_cla_feature_in=a_cla_feature_in)
    elif a_encoder == "trillsson3":
        audio_encoder = AudioFeatureProjection(input_feature_dim=1280, a_cla_feature_in=a_cla_feature_in)
    else:
        raise ValueError(f"Invalid audio encoder: {a_encoder}")
    return audio_encoder
