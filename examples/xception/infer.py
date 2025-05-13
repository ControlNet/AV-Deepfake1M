import argparse

import torch
from tqdm.auto import tqdm
from pathlib import Path

from avdeepfake1m.loader import AVDeepfake1mPlusPlusVideo
from xception import Xception

parser = argparse.ArgumentParser(description="Xception inference")
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--subset", type=str, choices=["train", "val", "test"])
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--take_num", type=int, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    use_gpu = args.gpus > 0
    device = "cuda" if use_gpu else "cpu"

    if args.model == "xception":
        model = Xception.load_from_checkpoint(args.checkpoint, lr=None, distributed=False).eval()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    model.train()
    test_dataset = AVDeepfake1mPlusPlusVideo(args.subset, args.data_root, take_num=args.take_num)

    save_path = f"output/{args.model}_{args.subset}.txt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        with torch.inference_mode():
            for i, (video, _, label) in enumerate(tqdm(test_dataset)):
                # batch video as frames use batch_size
                preds_video = []
                for j in range(0, len(video), args.batch_size):
                    batch = video[j:j + args.batch_size].to(device)
                    preds_video.append(model(batch))

                preds_video = torch.cat(preds_video, dim=0).flatten()
                # choose the max prediction
                pred = preds_video.max().item()

                file_name = test_dataset.metadata[i].file
                f.write(f"{file_name};{pred}\n")
