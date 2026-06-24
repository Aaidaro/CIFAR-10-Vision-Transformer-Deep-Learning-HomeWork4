# CIFAR-10 Vision Transformer Deep Learning Home Work 4

This repository implements a modular Vision Transformer (ViT) for CIFAR-10 using plain PyTorch modules. The main ViT is implemented from scratch: patch embedding, CLS token, positional embeddings, multi-head self-attention, Transformer blocks, training, evaluation, and attention visualization are all contained in this project.

## Project layout

```text
homework_code/
├── data/
│   ├── data_loader.py
│   └── dataset/
├── models/
│   ├── model.py
│   └── saved_models/
├── notebooks/
├── scripts/
│   ├── main.py
│   ├── train.py
│   ├── evaluate.py
│   ├── attention_map.py
│   ├── run_experiments.py
│   ├── compare_results.py
│   └── optional_pretrained_attention.py
├── config/
│   ├── config.yaml
│   └── logging.yaml
├── utils/
│   ├── visualization.py
│   ├── metrics.py
│   └── config.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Put your local CIFAR-10 file here:

```text
data/dataset/cifar-10-python.tar.gz
```

The loader will extract it automatically. You may also extract it manually so the project contains:

```text
data/dataset/cifar-10-batches-py/
```

## Quick sanity check

This runs the full pipeline on a small subset for two epochs:

```bash
python scripts/main.py train --case base --quick
```

The quick run writes checkpoints, plots, and metrics to:

```text
outputs/base_quick/
models/saved_models/base_quick_best.pt
```

## Part 2.1: base model training

Table 3 is encoded in `config/config.yaml`:

| Parameter | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Loss | Cross-entropy |
| Transformer layers | 4 |
| Attention heads | 8 |
| Embedding size | 128 |
| Patch size | 4x4 |
| Transformer MLP hidden width | 128 |

Train the base model:

```bash
python scripts/main.py train --case base
```

Useful overrides:

```bash
python scripts/main.py train --case base --epochs 150 --batch-size 128 --lr 0.0005
```

Outputs:

```text
outputs/base/base_loss.png
outputs/base/base_accuracy.png
outputs/base/history.csv
outputs/base/metrics.json
models/saved_models/base_best.pt
models/saved_models/base_last.pt
```

Evaluate a saved checkpoint:

```bash
python scripts/main.py evaluate --checkpoint models/saved_models/base_best.pt
```

## Part 2.2(a): run cases individually

Each model can be run independently:

```bash
python scripts/main.py train --case emb64
python scripts/main.py train --case emb256
python scripts/main.py train --case depth2
python scripts/main.py train --case depth8
python scripts/main.py train --case patch2
python scripts/main.py train --case patch8
python scripts/main.py train --case resize64
```

Or all cases sequentially:

```bash
python scripts/main.py train-many --cases base emb64 emb256 depth2 depth8 patch2 patch8 resize64
```

After several runs finish, create a comparison CSV:

```bash
python scripts/main.py compare
```

This creates:

```text
outputs/comparison_summary.csv
```

The summary includes trainable parameters, estimated MACs per image, validation accuracy, test accuracy, test loss, and elapsed seconds.

## Part 2.2(b): CLS attention maps

For the base model, visualize layers 1, 2, and 4:

```bash
python scripts/main.py attention \
  --checkpoint models/saved_models/base_best.pt \
  --layers 1 2 4 \
  --target-class 3
```

You can choose a specific test image index instead:

```bash
python scripts/main.py attention \
  --checkpoint models/saved_models/base_best.pt \
  --layers 1 2 4 \
  --index 25
```

The saved figure follows the format in the question: patched input, first attention head, and mean attention over heads for every requested layer.

## Part 2.2(c): pretrained `torchvision` ViT-B/16

The main assignment implementation does not use torchvision's ViT. This optional script is separate because the question explicitly permits pretrained `vit_b_16` for comparison.

```bash
python scripts/main.py optional-pretrained-attention --layers 4 8 12 --target-class 3
```



```bash
python scripts/main.py train --case base --epochs 180 --lr 0.0007
```

For fair comparison in Part 2.2(a), keep the optimizer, augmentation, seed, train/validation split, and epoch count the same across cases unless you explicitly report the difference.
