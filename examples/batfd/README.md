# BA-TFD

This example trains a Xception model on the AVDeepfake1M/AVDeepfake1M++ dataset for classification with video-level labels.
## Requirements

Ensure you have the necessary environment setup. You can create a Conda environment using the following commands:

```bash
# prepare the environment
conda create -n batfd python=3.10 -y
conda activate batfd
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install avdeepfake1m toml tensorboard pytorch-lightning pandas
```

## Training

Train the BATFD or BATFD+ model using a TOML configuration file (e.g., `batfd.toml` or `batfd_plus.toml`).

```bash
python train.py --config ./batfd.toml --data_root /path/to/AV-Deepfake1M-PlusPlus
```

### Output

*   **Checkpoints:** Model checkpoints are saved under `./ckpt/xception/`. The last checkpoint is saved as `last.ckpt`.
*   **Logs:** Training logs (including metrics like `train_loss`, `val_loss`, and learning rates) are saved by PyTorch Lightning, typically in a directory named `./lightning_logs/`. You can view these logs using TensorBoard (`tensorboard --logdir ./lightning_logs`). 

## Inference

After training, generate predictions on a dataset subset (e.g., `val`, `test`) using `infer.py`. This script saves the predictions to a JSON file, which is required for evaluation.

```bash
python infer.py --config ./batfd.toml --checkpoint /path/to/checkpoint --data_root /path/to/AV-Deepfake1M-PlusPlus --subset val
```

## Evaluation

```bash
python evaluate.py /path/to/prediction_file /path/to/metadata_file
```

