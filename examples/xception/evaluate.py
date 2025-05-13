import argparse

from avdeepfake1m.evaluation import auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for AV-Deepfake1M")
    parser.add_argument("prediction_file_path", type=str, help="Path to the prediction file (e.g., output/results/xception_val.txt)")
    parser.add_argument("metadata_file_path", type=str, help="Path to the metadata JSON file (e.g., /path/to/val_metadata.json)")
    args = parser.parse_args()

    print(auc(
        args.prediction_file_path,
        args.metadata_file_path,
        "file",  # As per README, this is usually "file"
        "fake_segments"  # As per README, this is usually "fake_segments" for AUC
    ))