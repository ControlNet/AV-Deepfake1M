import pandas as pd
import torch
from torchmetrics import AUROC

from ..utils import read_json


def auc(prediction_file: str, reference_path: str, file_key: str, value_key: str) -> float:

    prediction = pd.read_csv(prediction_file, header=None, sep=";")
    # convert to dict
    prediction_dict = {}
    for i in range(len(prediction)):
        prediction_dict[prediction.iloc[i, 0]] = prediction.iloc[i, 1]

    gt = read_json(reference_path)

    # make it as list
    truth = []
    prediction = []
    for gt_item in gt:
        key = gt_item[file_key]
        truth.append(int(len(gt_item[value_key]) > 0))
        prediction.append(prediction_dict[key])

    # to tensor for torchmetrics
    truth = torch.tensor(truth)
    prediction = torch.tensor(prediction)

    # compute auc
    auroc = AUROC(task="binary")
    return auroc(prediction, truth).item()
