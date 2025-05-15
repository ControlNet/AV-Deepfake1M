import os.path
from typing import Any, List, Optional
from torch import Tensor
import pandas as pd
from pathlib import Path
from lightning.pytorch import LightningModule, Trainer, Callback

from avdeepfake1m.loader import Metadata
from torch.utils.data import DataLoader


def nullable_index(obj, index):
    if obj is None:
        return None
    return obj[index]


class SaveToCsvCallback(Callback):

    def __init__(self, max_duration: int, metadata: List[Metadata], model_name: str, model_type: str, temp_dir: str):
        super().__init__()
        self.max_duration = max_duration
        self.metadata = metadata
        self.model_name = model_name
        self.model_type = model_type
        self.temp_dir = temp_dir

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.model_type == "batfd":
            fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla = outputs
            batch_size = fusion_bm_map.shape[0]

            for i in range(batch_size):
                temporal_size = batch[3][i]
                video_name = self.metadata[batch_idx * batch_size + i].file
                n_frames = self.metadata[batch_idx * batch_size + i].video_frames

                assert isinstance(video_name, str)
                self.gen_df_for_batfd(fusion_bm_map[i], temporal_size, n_frames, os.path.join(
                    self.temp_dir, self.model_name, video_name.replace("/", "_").replace(".mp4", ".csv")
                ))

        elif self.model_type == "batfd_plus":
            fusion_bm_map, fusion_start, fusion_end, v_bm_map, v_start, v_end, a_bm_map, a_start, a_end, v_frame_cla, a_frame_cla = outputs
            batch_size = fusion_bm_map.shape[0]

            for i in range(batch_size):
                temporal_size = batch[3][i]
                video_name = self.metadata[batch_idx * batch_size + i].file
                n_frames = self.metadata[batch_idx * batch_size + i].video_frames
                assert isinstance(video_name, str)

                self.gen_df_for_batfd_plus(fusion_bm_map[i], nullable_index(fusion_start, i),
                    nullable_index(fusion_end, i), temporal_size,
                    n_frames, os.path.join(self.temp_dir, self.model_name,
                        video_name.replace("/", "_").replace(".mp4", ".csv")
                    ))

        else:
            raise ValueError("Invalid model type")

    def gen_df_for_batfd(self, bm_map: Tensor, temporal_size: Tensor, n_frames: int, output_file: str):
        bm_map = bm_map.cpu().numpy()
        temporal_size = temporal_size.cpu().numpy().item()
        # for each boundary proposal in boundary map
        df = pd.DataFrame(bm_map)
        df = df.stack().reset_index()
        df.columns = ["duration", "begin", "score"]
        df["end"] = df.duration + df.begin
        df = df[(df.duration > 0) & (df.end <= temporal_size)]
        df = df.sort_values(["begin", "end"])
        df = df.reset_index()[["begin", "end", "score"]]
        df["begin"] = (df["begin"] / temporal_size * n_frames).astype(int)
        df["end"] = (df["end"] / temporal_size * n_frames).astype(int)
        df = df.sort_values(["score"], ascending=False).iloc[:100]
        df.to_csv(output_file, index=False)

    def gen_df_for_batfd_plus(self, bm_map: Tensor, start: Optional[Tensor], end: Optional[Tensor],
        temporal_size: Tensor, n_frames: int, output_file: str
    ):
        bm_map = bm_map.cpu().numpy()
        temporal_size = temporal_size.cpu().numpy().item()
        if start is not None and end is not None:
            start = start.cpu().numpy()
            end = end.cpu().numpy()

        # for each boundary proposal in boundary map
        df = pd.DataFrame(bm_map)
        df = df.stack().reset_index()
        df.columns = ["duration", "begin", "score"]
        df["end"] = df.duration + df.begin
        df = df[(df.duration > 0) & (df.end <= temporal_size)]
        df = df.sort_values(["begin", "end"])
        df = df.reset_index()[["begin", "end", "score"]]
        if start is not None and end is not None:
            df["score"] = df["score"] * start[df.begin] * end[df.end]

        df["begin"] = (df["begin"] / temporal_size * n_frames).astype(int)
        df["end"] = (df["end"] / temporal_size * n_frames).astype(int)
        df = df.sort_values(["score"], ascending=False).iloc[:100]
        df.to_csv(output_file, index=False)


def inference_model(model_name: str, model: LightningModule, dataloader: DataLoader,
    metadata: List[Metadata],
    max_duration: int, model_type: str,
    gpus: int = 1,
    temp_dir: str = "output/",
    subset: str = "test"
) -> List[Metadata]:
    Path(os.path.join(temp_dir, model_name)).mkdir(parents=True, exist_ok=True)
    assert subset in ["test", "val"]

    model.eval()

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=1 if gpus > 1 else "auto",
        accelerator="auto" if gpus > 0 else "cpu",
        callbacks=[SaveToCsvCallback(max_duration, metadata, model_name, model_type, temp_dir)]
    )

    trainer.predict(model, dataloader)
