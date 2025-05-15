import os
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Union, Tuple

import numpy as np
import torch
import torchaudio
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler, IterableDataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import read_json, read_video, resize_video, iou_with_anchors, \
    read_video_fast, read_video, iou_1d


@dataclass
class Metadata:
    file: str
    original: Optional[str]
    split: str
    n_fakes: int
    duration: float
    fake_periods: List[List[float]]
    modify_type: str
    modify_video: bool
    modify_audio: bool
    audio_model: str
    video_frames: int
    audio_frames: int

    def __init__(self, file: str, original: Optional[str], split: str, fake_segments: List[List[float]], fps: int,
        visual_fake_segments: List[List[float]], audio_fake_segments: List[List[float]],
        audio_model: str, modify_type: str, video_frames: int, audio_frames: int, *args, **kwargs
    ):
        self.file = file
        self.original = original
        self.split = split
        self.n_fakes = len(fake_segments)
        self.duration = video_frames / fps
        self.fake_periods = fake_segments
        self.visual_fake_periods = visual_fake_segments
        self.audio_fake_periods = audio_fake_segments
        self.modify_type = modify_type
        self.modify_video = modify_type in ("both-modified", "visual_modified")
        self.modify_audio = modify_type in ("both-modified", "audio_modified")
        self.audio_model = audio_model
        self.video_frames = video_frames
        self.audio_frames = audio_frames


T_LABEL = Union[Tensor, Tuple[Tensor, Tensor, Tensor]]


class AVDeepfake1m(Dataset):

    def __init__(self, subset: str, data_root: str = "data", temporal_size: int = 100,
        max_duration: int = 30, fps: int = 25,
        video_transform: Callable[[Tensor], Tensor] = Identity(),
        audio_transform: Callable[[Tensor], Tensor] = Identity(),
        file_list: Optional[List[str]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, T_LABEL], List[Any]] = None,
        require_match_scores: bool = False,
        return_file_name: bool = False,
        is_plusplus: bool = False
    ):
        self.subset = subset
        self.root = data_root
        self.fps = fps
        self.temporal_size = temporal_size
        self.audio_temporal_size = int(temporal_size / fps * 16000)
        self.max_duration = max_duration
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.get_meta_attr = get_meta_attr
        self.require_match_scores = require_match_scores
        self.return_file_name = return_file_name
        self.is_plusplus = is_plusplus  # For AV-Deepfake1M++, we modify the structure a little bit.

        label_dir = os.path.join(self.root, "label")
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        if file_list is None:
            self.file_list = [meta["file"] for meta in read_json(os.path.join(self.root, f"{subset}_metadata.json"))]
        else:
            self.file_list = file_list

        if self.require_match_scores:
            temporal_gap = 1 / self.max_duration
            # [-0.05, ..., 0.985]
            self.anchor_x_min = [temporal_gap * (i - 0.5) for i in range(self.temporal_size)]
            # [0.05, ..., 0.995]
            self.anchor_x_max = [temporal_gap * (i + 0.5) for i in range(self.temporal_size)]
        else:
            self.anchor_x_min = None
            self.anchor_x_max = None

        print(f"Load {len(self.file_list)} data in {subset}.")

    def __getitem__(self, index: int) -> List[Union[Tensor, str, int]]:
        file = self.file_list[index]

        video, audio, _ = read_video(os.path.join(self.root, self.subset, file))
        video = F.interpolate(video.float().permute(1, 0, 2, 3)[None], size=(self.temporal_size, 96, 96))[0]
        audio = F.interpolate(audio.float().permute(1, 0)[None], size=self.audio_temporal_size, mode="linear")[0].permute(1, 0)
        video = self.video_transform(video)
        audio = self.audio_transform(audio)
        audio = self._get_log_mel_spectrogram(audio)

        outputs = [video, audio]

        if self.subset != "test":
            if self.is_plusplus:
                subset_folder = self.subset
            else:
                subset_folder = self.subset + "_metadata"
            meta = read_json(os.path.join(self.root, subset_folder, file.replace(".mp4", ".json")))
            meta = Metadata(**meta, fps=self.fps)
            if not self.require_match_scores:
                label, visual_label, audio_label = self.get_label(file, meta)
                outputs = outputs + [label] + self.get_meta_attr(meta, video, audio, (label, visual_label, audio_label))
            else:
                label, visual_label, audio_label = self.get_label(file, meta)
                outputs = outputs + [label, 0, 0] + self.get_meta_attr(meta, video, audio,
                    (label, visual_label, audio_label))

            if self.return_file_name:
                outputs.append(meta.file)

        return outputs

    def get_label(self, file: str, meta: Metadata) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        file_name = file.replace("/", "_").split(".")[0] + ".npz"
        path = os.path.join(self.root, "label", file_name)
        if os.path.exists(path):
            try:
                npz = np.load(path)
            except ValueError:
                pass
            except EOFError:
                pass
            else:
                return (
                    torch.from_numpy(npz["label"]),
                    torch.from_numpy(npz["visual_label"]) if npz["visual_label"].shape != () else None,
                    torch.from_numpy(npz["audio_label"]) if npz["audio_label"].shape != () else None
                )

        label_obj = {
            "label": -1,
            "visual_label": -1,
            "audio_label": -1
        }
        label_obj["label"] = self._get_train_label(meta.video_frames, meta.fake_periods, self.temporal_size).numpy()

        if len(meta.visual_fake_periods) > 0 and meta.visual_fake_periods != meta.fake_periods:
            label_obj["visual_label"] = self._get_train_label(meta.video_frames, meta.visual_fake_periods,
                self.temporal_size).numpy()

        if len(meta.audio_fake_periods) > 0 and meta.audio_fake_periods != meta.fake_periods:
            label_obj["audio_label"] = self._get_train_label(meta.video_frames, meta.audio_fake_periods,
                self.temporal_size).numpy()

        # cache label
        np.savez(path, **label_obj)
        assert type(label_obj["label"]) == np.ndarray

        return (torch.from_numpy(label_obj["label"]),
        torch.from_numpy(label_obj["visual_label"]) if type(label_obj["visual_label"]) == np.ndarray else None,
        torch.from_numpy(label_obj["audio_label"]) if type(label_obj["audio_label"]) == np.ndarray else None)

    def gen_label(self) -> None:
        # manually pre-generate label as npy
        for file in tqdm(self.file_list):
            meta = read_json(os.path.join(self.root, "data", file.replace(".mp4", ".json")))
            meta = Metadata(**meta)
            self.get_label(file, meta)

    def __len__(self) -> int:
        return len(self.file_list)

    def _get_log_mel_spectrogram(self, audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        assert spec.shape == (64, 4 * self.temporal_size), "Wrong log mel-spectrogram setup in Dataset"
        return spec

    def _get_train_label(self, frames, video_labels, temporal_scale, fps=25) -> Tensor:
        corrected_second = frames / fps
        temporal_gap = 1 / temporal_scale

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_start = max(min(1, video_labels[j][0] / corrected_second), 0)
            tmp_end = max(min(1, video_labels[j][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = torch.tensor(gt_bbox)
        if len(gt_bbox) > 0:
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
        else:
            gt_xmins = np.array([])
            gt_xmaxs = np.array([])
        #####################################################################################################

        gt_iou_map = torch.zeros([self.max_duration, temporal_scale])
        if len(gt_bbox) > 0:
            for begin in range(temporal_scale):
                for duration in range(self.max_duration):
                    end = begin + duration
                    if end > temporal_scale:
                        break
                    gt_iou_map[duration, begin] = torch.max(
                        iou_with_anchors(begin * temporal_gap, (end + 1) * temporal_gap, gt_xmins, gt_xmaxs))
                    # [i, j]: Start in i, end in j.

        ##########################################################################################################
        return gt_iou_map


def _default_get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor) -> List[Any]:
    return [meta.video_frames]


class AVDeepfake1mDataModule(LightningDataModule):
    train_dataset: AVDeepfake1m
    val_dataset: AVDeepfake1m
    test_dataset: AVDeepfake1m

    def __init__(self, root: str = "data", temporal_size: int = 100,
        max_duration: int = 30, fps: int = 25,
        require_match_scores: bool = False,
        batch_size: int = 1, num_workers: int = 0,
        take_train: Optional[int] = None, take_val: Optional[int] = None, take_test: Optional[int] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = _default_get_meta_attr,
        return_file_name: bool = False,
        is_plusplus: bool = False,
    ):
        super().__init__()
        self.root = root
        self.temporal_size = temporal_size
        self.max_duration = max_duration
        self.fps = fps
        self.require_match_scores = require_match_scores
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test
        self.get_meta_attr = get_meta_attr
        self.return_file_name = return_file_name
        self.is_plusplus = is_plusplus
        self.Dataset = AVDeepfake1m

    def setup(self, stage: Optional[str] = None) -> None:
        train_file_list = [meta["file"] for meta in read_json(os.path.join(self.root, "train_metadata.json"))]
        val_file_list = [meta["file"] for meta in read_json(os.path.join(self.root, "val_metadata.json"))]
        with open(os.path.join(self.root, "test_files.txt"), "r") as f:
            test_file_list = list(filter(lambda x: x != "", f.read().split("\n")))

        # take subset of data if specified
        if self.take_train is not None:
            train_file_list = train_file_list[:self.take_train]

        if self.take_val is not None:
            val_file_list = val_file_list[:self.take_val]

        if self.take_test is not None:
            test_file_list = test_file_list[:self.take_test]

        self.train_dataset = self.Dataset("train", self.root, self.temporal_size, self.max_duration, self.fps,
            file_list=train_file_list, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores,
            return_file_name=self.return_file_name,
            is_plusplus=self.is_plusplus
        )
        self.val_dataset = self.Dataset("val", self.root, self.temporal_size, self.max_duration, self.fps,
            file_list=val_file_list, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores,
            return_file_name=self.return_file_name,
            is_plusplus=self.is_plusplus
        )
        self.test_dataset = self.Dataset("test", self.root, self.temporal_size, self.max_duration, self.fps,
            file_list=test_file_list, get_meta_attr=self.get_meta_attr,
            require_match_scores=self.require_match_scores,
            return_file_name=self.return_file_name,
            is_plusplus=self.is_plusplus
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            sampler=RandomSampler(self.train_dataset, num_samples=self.take_train, replacement=True),
            drop_last=True, pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)


class AVDeepfake1mImages(IterableDataset):

    def __init__(self, subset: str, data_root: str = "data",
        image_size: int = 96,
        use_video_label: bool = False,
        use_seg_label: Optional[int] = None,
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):
        self.subset = subset
        self.data_root = data_root
        self.image_size = image_size
        self.use_video_label = use_video_label
        if self.use_video_label:
            assert use_seg_label is None
        self.use_seg_label = use_seg_label
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[:take_num]

        self.total_frames = sum([each.video_frames for each in self.metadata])
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        for meta in self.metadata:
            video = read_video_fast(os.path.join(self.data_root, "data", meta.file))
            if self.image_size != 224:
                video = resize_video(video, (96, 96))
            if self.use_video_label:
                label = float(len(meta.fake_periods) > 0)
                for frame in video:
                    yield frame, label
            elif self.use_seg_label:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25)
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                seg_label = torch.split(frame_label, self.use_seg_label)
                seg_label = torch.nn.utils.rnn.pad_sequence(seg_label, batch_first=True)
                seg_label = (seg_label.sum(dim=1) > 0).float().repeat_interleave(self.use_seg_label)
                for i, frame in enumerate(video):
                    yield frame, seg_label[i]
            else:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25)
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]


class AVDeepfake1mVideo(Dataset):

    def __init__(self, subset: str, data_root: str = "data",
        image_size: int = 96,
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):
        self.subset = subset
        self.data_root = data_root
        self.image_size = image_size
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata[index]
        video, audio, _ = read_video(os.path.join(self.data_root, "data", meta.file))
        if self.image_size != 224:
            video = resize_video(video, (self.image_size, self.image_size))
        label = len(meta.fake_periods) > 0
        return video, audio, label


class AVDeepfake1mAudio(Dataset):

    def __init__(self, subset: str, data_root: str = "data",
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):

        self.subset = subset
        self.data_root = data_root
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata[index]
        audio, _ = torchaudio.load(os.path.join(self.data_root, "data", meta.file))
        label = len(meta.fake_periods) > 0
        return audio, label


class AVDeepfake1mSegment(Dataset):

    def __init__(self, subset: str, data_root: str = "data",
        frame_size: int = 96,
        segment_length: int = 16,
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):
        super().__init__()
        self.subset = subset
        self.data_root = data_root
        self.frame_size = frame_size
        self.segment_length = segment_length
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[::int(len(self.metadata) // take_num)]

        self.mfcc_fn = torchaudio.transforms.MFCC(sample_rate=16000, melkwargs={"n_fft": 2048})
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata[index]
        fps = 25
        video, audio, _ = read_video(os.path.join(self.data_root, "data", meta.file))
        if self.frame_size != 224:
            video = resize_video(video, (self.frame_size, self.frame_size))

        indexes = sample_indexes(meta.video_frames, self.segment_length, 2)

        if len(meta.fake_periods) == 0:
            label = torch.tensor(0.0)
        else:
            ious = iou_1d((indexes / fps)[None, [0, -1]], torch.tensor(meta.fake_periods))
            label = (ious > 0).any().to(torch.float32)

        if indexes.max() >= video.shape[0]:
            indexes = indexes - (indexes.max() - video.shape[0] + 1)
        segment = video[indexes].permute(1, 0, 2, 3)

        return segment, label


def sample_indexes(total_frames: int, n_frames: int, temporal_sample_rate: int):
    try:
        start_ind = torch.randint(0, total_frames - (n_frames * temporal_sample_rate), ())
    except RuntimeError as e:
        print(f"total_frames: {total_frames}, n_frames: {n_frames}, temporal_sample_rate: {temporal_sample_rate}")
        raise e
    return torch.arange(n_frames) * temporal_sample_rate + start_ind


class AVDeepfake1mPlusPlusImages(IterableDataset):

    def __init__(self, subset: str, data_root: str = "data",
        image_size: int = 96,
        use_video_label: bool = False,
        use_seg_label: Optional[int] = None,
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):
        self.subset = subset
        self.data_root = data_root
        self.image_size = image_size
        self.use_video_label = use_video_label
        if self.use_video_label:
            assert use_seg_label is None
        self.use_seg_label = use_seg_label
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[:take_num]

        self.total_frames = sum([each.video_frames for each in self.metadata])
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return self.total_frames

    def __iter__(self):
        for meta in self.metadata:
            video = read_video_fast(os.path.join(self.data_root, self.subset, meta.file))
            if self.image_size != 224:
                video = resize_video(video, (96, 96))
            if self.use_video_label:
                label = float(len(meta.fake_periods) > 0)
                for frame in video:
                    yield frame, label
            elif self.use_seg_label:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25)
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                seg_label = torch.split(frame_label, self.use_seg_label)
                seg_label = torch.nn.utils.rnn.pad_sequence(seg_label, batch_first=True)
                seg_label = (seg_label.sum(dim=1) > 0).float().repeat_interleave(self.use_seg_label)
                for i, frame in enumerate(video):
                    yield frame, seg_label[i]
            else:
                frame_label = torch.zeros(len(video))
                for begin, end in meta.fake_periods:
                    begin = int(begin * 25)
                    end = int(end * 25)
                    frame_label[begin: end] = 1
                for i, frame in enumerate(video):
                    yield frame, frame_label[i]


class AVDeepfake1mPlusPlusVideo(Dataset):

    def __init__(self, subset: str, data_root: str = "data",
        image_size: int = 96,
        take_num: Optional[int] = None,
        metadata: Optional[List[Metadata]] = None,
    ):
        self.subset = subset
        self.data_root = data_root
        self.image_size = image_size
        if metadata is None:
            metadata_json = read_json(os.path.join(self.data_root, f"{subset}_metadata.json"))
            self.metadata = [Metadata(**meta, fps=25) for meta in metadata_json]
        else:
            self.metadata = metadata

        if take_num is not None:
            self.metadata = self.metadata[:take_num]
        print("Load {} data in {}.".format(len(self.metadata), subset))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata[index]
        video, audio, _ = read_video(os.path.join(self.data_root, self.subset, meta.file))
        if self.image_size != 224:
            video = resize_video(video, (self.image_size, self.image_size))
        label = len(meta.fake_periods) > 0
        return video, audio, label
