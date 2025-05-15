import json
import os.path
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from avdeepfake1m.loader import Metadata
from avdeepfake1m.utils import iou_with_anchors


def soft_nms(df, alpha, t1, t2, fps):
    df = df.sort_values(by="score", ascending=False)
    t_start = list(df.begin.values[:] / fps)
    t_end = list(df.end.values[:] / fps)
    t_score = list(df.score.values[:])

    r_start = []
    r_end = []
    r_score = []

    while len(t_score) > 1 and len(r_score) < 101:
        max_index = t_score.index(max(t_score))
        tmp_iou_list = iou_with_anchors(
            np.array(t_start),
            np.array(t_end), t_start[max_index], t_end[max_index])
        for idx in range(0, len(t_score)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = t_end[max_index] - t_start[max_index]
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    t_score[idx] *= np.exp(-np.square(tmp_iou) / alpha)

        r_start.append(t_start[max_index])
        r_end.append(t_end[max_index])
        r_score.append(t_score[max_index])
        t_start.pop(max_index)
        t_end.pop(max_index)
        t_score.pop(max_index)

    new_df = pd.DataFrame()
    new_df['score'] = r_score
    new_df['begin'] = r_start
    new_df['end'] = r_end
    return new_df


def video_post_process(meta, model_name, fps=25, alpha=0.4, t1=0.2, t2=0.9, dataset_name="avdeepfake1m", output_dir="output"):
    file = resolve_csv_file_name(meta, dataset_name)
    df = pd.read_csv(os.path.join(output_dir, model_name, file))

    if len(df) > 1:
        df = soft_nms(df, alpha, t1, t2, fps)

    df = df.sort_values(by="score", ascending=False)

    proposal_list = []

    for j in range(len(df)):
        proposal_list.append([
            df.score.values[j],
            df.begin.values[j].item(),
            df.end.values[j].item()
        ])

    return [meta.file, proposal_list]


def resolve_csv_file_name(meta: Metadata, dataset_name: str = "avdeepfake1m") -> str:
    if dataset_name in ("avdeepfake1m", "avdeepfake1m++"):
        return meta.file.replace("/", "_").replace(".mp4", ".csv")
    else:
        raise NotImplementedError


def post_process(model_name: str, save_path: str, metadata: List[Metadata], fps=25,
    alpha=0.4, t1=0.2, t2=0.9, dataset_name="avdeepfake1m", output_dir="output"
):
    with ProcessPoolExecutor(cpu_count() // 2 - 1) as executor:
        futures = []
        for meta in metadata:
            futures.append(executor.submit(video_post_process, meta, model_name, fps,
                alpha, t1, t2, dataset_name, output_dir
            ))

        results = dict(map(lambda x: x.result(), tqdm(futures)))

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
