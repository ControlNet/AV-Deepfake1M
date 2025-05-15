import argparse
import json
import math

from avdeepfake1m.evaluation import ap_ar_1d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for BATFD/BATFD+ models on AV-Deepfake1M")
    parser.add_argument("prediction_file_path", type=str, help="Path to the prediction JSON file (e.g., output/batfd_test.json)")
    parser.add_argument("metadata_file_path", type=str, help="Path to the metadata JSON file (e.g., /path/to/dataset/test_metadata.json or /path/to/dataset/val_metadata.json)")
    args = parser.parse_args()

    print(f"Calculating AP/AR for prediction file: {args.prediction_file_path}")
    print(f"Using metadata file: {args.metadata_file_path}")

    # Parameters for ap_ar_1d based on README.md
    file_key = "file"
    value_key = "fake_segments" # For ground truth in metadata
    fps = 1.0 
    ap_iou_thresholds = [0.5, 0.75, 0.9, 0.95]
    ar_n_proposals = [50, 30, 20, 10, 5]
    ar_iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ap_ar_results = ap_ar_1d(
        proposals_path=args.prediction_file_path,
        labels_path=args.metadata_file_path,
        file_key=file_key,
        value_key=value_key,
        fps=fps,
        ap_iou_thresholds=ap_iou_thresholds,
        ar_n_proposals=ar_n_proposals,
        ar_iou_thresholds=ar_iou_thresholds
    )

    print(ap_ar_results)

    score = 0.5 * sum(ap_ar_results["ap"].values()) / len(ap_ar_results["ap"]) \
        + 0.5 * sum(ap_ar_results["ar"].values()) / len(ap_ar_results["ar"])

    print(f"Score: {score}")
