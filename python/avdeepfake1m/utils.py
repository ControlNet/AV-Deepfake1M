import json
import os
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

def read_json(path: str, object_hook=None):
    """Read a JSON file and return the parsed data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)

def read_video(path: str, return_numpy: bool = False) -> Tuple[Tensor, Tensor, dict]:
    """Read a video file and return video frames, audio, and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2) / 255  # Convert to (T, C, H, W)
    audio = audio.permute(1, 0)  # Rearrange audio tensor

    if return_numpy:
        return video.numpy(), audio.numpy(), info
    
    return video, audio, info

def read_videos(paths: List[str]) -> List[Tuple[Tensor, Tensor, dict]]:
    """Read multiple videos and return a list of their frames, audio, and metadata."""
    videos = []
    for path in tqdm(paths, desc="Reading videos"):
        videos.append(read_video(path))
    return videos

def read_video_fast(path: str) -> Tensor:
    """Read a video quickly using OpenCV and return the frames as a tensor."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    video = np.stack(frames, axis=0)
    video = rearrange(video, 'T H W C -> T C H W')
    return torch.from_numpy(video) / 255

def resize_video(tensor: Tensor, size: Tuple[int, int], resize_method: str = "bicubic") -> Tensor:
    """Resize the input tensor to the specified size using the given method."""
    return F.interpolate(tensor, size=size, mode=resize_method)

def iou_with_anchors(anchors_min: np.ndarray, anchors_max: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Compute Jaccard score (IoU) between a box and the anchors."""
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    union_len = len_anchors - inter_len + (box_max - box_min)
    iou = inter_len / union_len
    return iou

def ioa_with_anchors(anchors_min: np.ndarray, anchors_max: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Calculate the overlap proportion (IoA) between the anchor and a bounding box."""
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    scores = np.divide(inter_len, len_anchors)
    return scores

def iou_1d(proposal: Union[Tensor, np.ndarray], target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Calculate 1D IoU for N proposals with L targets.

    Args:
        proposal: Predicted array with shape [M, 2].
        target: Label array with shape [N, 2].

    Returns:
        Tensor: IoU results with shape [M, N].
    """
    if isinstance(proposal, np.ndarray):
        proposal = torch.from_numpy(proposal)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.0)
    union = outer_end - outer_begin
    return inter / union
