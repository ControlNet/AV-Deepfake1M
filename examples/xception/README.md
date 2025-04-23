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

## Output

*   **Checkpoints:** Model checkpoints are saved under `./ckpt1/xception/`. The last checkpoint is saved as `last.ckpt`.
*   **Logs:** Training logs (including metrics like `train_loss`, `val_loss`, and learning rates) are saved by PyTorch Lightning, typically in a directory named `./lightning_logs/`. You can view these logs using TensorBoard (`tensorboard --logdir ./lightning_logs`). 
