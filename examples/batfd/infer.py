import argparse
import toml
import torch
import os
from pathlib import Path

from avdeepfake1m.loader import AVDeepfake1mDataModule, Metadata
from batfd.model import Batfd, BatfdPlus
from batfd.inference import inference_model
from batfd.post_process import post_process
from avdeepfake1m.utils import read_json

def main():
    parser = argparse.ArgumentParser(description="BATFD/BATFD+ Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the TOML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading.")
    parser.add_argument("--subset", type=str, choices=["val", "test"], 
                        default="test", help="Dataset subset.")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs. Set to 0 for CPU.")

    args = parser.parse_args()

    # Determine device
    if args.gpus > 0 and torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    config = toml.load(args.config)
    temp_dir = "output"
    output_file = os.path.join(temp_dir, f"{config['name']}_{args.subset}.json")
    model_type = config["model_type"]

    if model_type == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(args.checkpoint)
    elif model_type == "batfd":
        model = Batfd.load_from_checkpoint(args.checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()

    # Setup DataModule
    dm_dataset_name = config["dataset"]
    is_plusplus = dm_dataset_name == "avdeepfake1m++"

    dm = AVDeepfake1mDataModule(
        root=args.data_root,
        temporal_size=config["num_frames"],
        max_duration=config["max_duration"],
        require_match_scores=False,
        batch_size=1, # due to the problem from lightning, only 1 is supported
        num_workers=args.num_workers,
        get_meta_attr=model.get_meta_attr,
        return_file_name=True,
        is_plusplus=is_plusplus
    )
    dm.setup()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    if args.subset == "test":
        dataloader = dm.test_dataloader()
        metadata_path = os.path.join(dm.root, "test_metadata.json")
    elif args.subset == "val":
        dataloader = dm.val_dataloader()
        metadata_path = os.path.join(dm.root, "val_metadata.json")
    else:
        raise ValueError("Invalid subset")

    metadata = [Metadata(**each, fps=25) for each in read_json(metadata_path)]

    inference_model(
        model_name=config["name"], 
        model=model, 
        dataloader=dataloader,
        metadata=metadata,
        max_duration=config["max_duration"], 
        model_type=config["model_type"], 
        gpus=args.gpus, 
        temp_dir=temp_dir, 
        subset=args.subset
    )

    post_process(
        model_name=config["name"], 
        save_path=output_file,
        metadata=metadata, 
        fps=25, 
        alpha=config["soft_nms"]["alpha"], 
        t1=config["soft_nms"]["t1"], 
        t2=config["soft_nms"]["t2"], 
        dataset_name=dm_dataset_name,
        output_dir=temp_dir
    )

    print(f"Inference complete. Results saved to {output_file}")


if __name__ == '__main__':
    main()
