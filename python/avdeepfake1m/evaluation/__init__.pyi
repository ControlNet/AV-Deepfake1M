from typing import Dict, List, Union


def ap_1d(proposals_path: str, labels_path: str, file_key: str, value_key: str, fps: float,
          iou_thresholds: List[float]) -> Dict[float, float]:
    pass


def ar_1d(proposals_path: str, labels_path: str, file_key: str, value_key: str, fps: float, n_proposals: List[int],
          iou_thresholds: List[float]) -> Dict[int, float]:
    pass


def ap_ar_1d(
        proposals_path: str, labels_path: str, file_key: str, value_key: str, fps: float,
        ap_iou_thresholds: List[float], ar_n_proposals: List[int], ar_iou_thresholds: List[float]
) -> Dict[str, Dict[Union[float, int], float]]:
    pass


def auc(prediction_file: str, reference_path: str, file_key: str, value_key: str) -> float:
    pass
