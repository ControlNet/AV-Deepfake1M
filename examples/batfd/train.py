import argparse

import toml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from avdeepfake1m.loader import AVDeepfake1mDataModule
from batfd.model import Batfd, BatfdPlus
from batfd.utils import LrLogger, EarlyStoppingLR

parser = argparse.ArgumentParser(description="BATFD training")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=32)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=1000)
parser.add_argument("--max_epochs", type=int, default=500)
parser.add_argument("--resume", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)

    learning_rate = config["optimizer"]["learning_rate"]
    gpus = args.gpus
    total_batch_size = args.batch_size * gpus
    learning_rate = learning_rate * total_batch_size / 4
    dataset = config["dataset"]

    v_encoder_type = config["model"]["video_encoder"]["type"]
    a_encoder_type = config["model"]["audio_encoder"]["type"]

    if v_encoder_type in ("marlin_vit_small", "3dmm", "i3d"):
        v_feature = v_encoder_type
    else:
        v_feature = None

    if a_encoder_type in ("deep_speech", "wav2vec2", "trill"):
        a_feature = a_encoder_type
    else:
        a_feature = None

    if config["model_type"] == "batfd_plus":
        model = BatfdPlus(
            v_encoder=v_encoder_type,
            a_encoder=config["model"]["audio_encoder"]["type"],
            frame_classifier=config["model"]["frame_classifier"]["type"],
            ve_features=config["model"]["video_encoder"]["hidden_dims"],
            ae_features=config["model"]["audio_encoder"]["hidden_dims"],
            v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
            a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
            boundary_features=config["model"]["boundary_module"]["hidden_dims"],
            boundary_samples=config["model"]["boundary_module"]["samples"],
            temporal_dim=config["num_frames"],
            max_duration=config["max_duration"],
            weight_frame_loss=config["optimizer"]["frame_loss_weight"],
            weight_modal_bm_loss=config["optimizer"]["modal_bm_loss_weight"],
            weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
            contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
            cbg_feature_weight=config["optimizer"]["cbg_feature_weight"],
            prb_weight_forward=config["optimizer"]["prb_weight_forward"],
            weight_decay=config["optimizer"]["weight_decay"],
            learning_rate=learning_rate,
            distributed=args.gpus > 1
        )
        require_match_scores = True
        get_meta_attr = BatfdPlus.get_meta_attr
    elif config["model_type"] == "batfd":
        model = Batfd(
            v_encoder=config["model"]["video_encoder"]["type"],
            a_encoder=config["model"]["audio_encoder"]["type"],
            frame_classifier=config["model"]["frame_classifier"]["type"],
            ve_features=config["model"]["video_encoder"]["hidden_dims"],
            ae_features=config["model"]["audio_encoder"]["hidden_dims"],
            v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
            a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
            boundary_features=config["model"]["boundary_module"]["hidden_dims"],
            boundary_samples=config["model"]["boundary_module"]["samples"],
            temporal_dim=config["num_frames"],
            max_duration=config["max_duration"],
            weight_frame_loss=config["optimizer"]["frame_loss_weight"],
            weight_modal_bm_loss=config["optimizer"]["modal_bm_loss_weight"],
            weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
            contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
            weight_decay=config["optimizer"]["weight_decay"],
            learning_rate=learning_rate,
            distributed=args.gpus > 1
        )
        require_match_scores = False
        get_meta_attr = Batfd.get_meta_attr
    else:
        raise ValueError("Invalid model type")
    
    if dataset == "avdeepfake1m":
        dm = AVDeepfake1mDataModule(
            root=args.data_root,
            temporal_size=config["num_frames"],
            max_duration=config["max_duration"],
            require_match_scores=require_match_scores,
            batch_size=args.batch_size, num_workers=args.num_workers,
            take_train=args.num_train, take_val=args.num_val,
            get_meta_attr=get_meta_attr,
            is_plusplus=False
        )
    elif dataset == "avdeepfake1m++":
        dm = AVDeepfake1mDataModule(
            root=args.data_root,
            temporal_size=config["num_frames"],
            max_duration=config["max_duration"],
            require_match_scores=require_match_scores,
            batch_size=args.batch_size, num_workers=args.num_workers,
            take_train=args.num_train, take_val=args.num_val,
            get_meta_attr=get_meta_attr,
            is_plusplus=True
        )
    else:
        raise ValueError("Invalid dataset type")

    try:
        precision = int(args.precision)
    except ValueError:
        precision: int | str = args.precision
        
    monitor = "val_loss"

    trainer = Trainer(log_every_n_steps=50, precision=precision, max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./ckpt/{config['name']}", save_last=True, filename=config["name"] + "-{epoch}-{val_loss:.3f}",
                monitor=monitor, mode="min"
            ),
            LrLogger(),
            EarlyStoppingLR(lr_threshold=1e-7)
        ], enable_checkpointing=True,
        benchmark=True,
        accelerator="auto",
        devices=args.gpus,
        strategy="auto" if args.gpus < 2 else "ddp",
    )

    torch.set_float32_matmul_precision('high')

    trainer.fit(model, dm, ckpt_path=args.resume)
