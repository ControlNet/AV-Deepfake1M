# Xception

This example trains a Xception model on the AVDeepfake1M/AVDeepfake1M++ dataset for classification with video-level labels.

## Requirements

- Python
- PyTorch
- PyTorch Lightning
- TIMM
- AVDeepfake1M SDK


## Training

```bash
python train.py --data_root /path/to/avdeepfake1m --model xception
```
### Output

*   **Checkpoints:** Model checkpoints are saved under `./ckpt/xception/`. The last checkpoint is saved as `last.ckpt`.
*   **Logs:** Training logs (including metrics like `train_loss`, `val_loss`, and learning rates) are saved by PyTorch Lightning, typically in a directory named `./lightning_logs/`. You can view these logs using TensorBoard (`tensorboard --logdir ./lightning_logs`). 


## Inference

After training, you can generate predictions on a dataset subset (train, val, or test) using `infer.py`. This script will save the predictions to a text file, following the format from the [challenge](https://deepfakes1m.github.io/2025/details).

```bash
python infer.py --data_root /path/to/avdeepfake1m --checkpoint /path/to/your/checkpoint.ckpt --model xception --subset val
```

The output prediction file will be saved to `output/<model_name>_<subset>.txt` (e.g., `output/xception_val.txt`).

## Evaluation

```bash
python evaluate.py <path_to_prediction_file> <path_to_metadata_json>
```

For example:

```bash
python evaluate.py ./output/xception_val.txt /path/to/avdeepfake1m/val_metadata.json
```

This will print the AUC score based on your model's predictions. 
