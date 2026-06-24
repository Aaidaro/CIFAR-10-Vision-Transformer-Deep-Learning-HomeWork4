"""Main command-line entry point for the modular ViT homework project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attention_map import attention_from_args
from scripts.compare_results import compare_results
from scripts.evaluate import evaluate_from_args
from scripts.optional_pretrained_attention import pretrained_vit_attention
from scripts.run_experiments import run_cases
from scripts.train import train_from_args


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by train-style commands."""
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--case", default="base", help="Named case from config.yaml, e.g. base, emb64, patch2.")
    parser.add_argument("--quick", action="store_true", help="Use tiny subsets and few epochs for a fast sanity check.")
    parser.add_argument("--data-dir", default=None, help="Relative path containing cifar-10-python.tar.gz or extracted batches.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--subset-train", type=int, default=None, help="Use only N training examples.")
    parser.add_argument("--subset-val", type=int, default=None, help="Use only N validation examples.")
    parser.add_argument("--subset-test", type=int, default=None, help="Use only N test examples.")


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(description="CIFAR-10 ViT homework runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one named model case.")
    _add_common_train_args(train_parser)

    all_parser = subparsers.add_parser("train-many", help="Train several cases sequentially.")
    all_parser.add_argument("--config", default="config/config.yaml")
    all_parser.add_argument("--cases", nargs="+", default=["base", "emb64", "emb256", "depth2", "depth8", "patch2", "patch8", "resize64"])
    all_parser.add_argument("--quick", action="store_true")
    all_parser.add_argument("--data-dir", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint on the test split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. models/saved_models/base_best.pt")
    eval_parser.add_argument("--config", default="config/config.yaml")
    eval_parser.add_argument("--case", default="base")
    eval_parser.add_argument("--quick", action="store_true")
    eval_parser.add_argument("--data-dir", default=None)
    eval_parser.add_argument("--subset-test", type=int, default=None)

    attn_parser = subparsers.add_parser("attention", help="Plot CLS attention maps for a trained checkpoint.")
    attn_parser.add_argument("--checkpoint", required=True)
    attn_parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4], help="1-indexed layers to visualize.")
    attn_parser.add_argument("--target-class", type=int, default=None, help="Choose the first test image with this class id.")
    attn_parser.add_argument("--index", type=int, default=None, help="Choose a specific test image index.")
    attn_parser.add_argument("--config", default="config/config.yaml")
    attn_parser.add_argument("--case", default="base")
    attn_parser.add_argument("--quick", action="store_true")
    attn_parser.add_argument("--data-dir", default=None)

    compare_parser = subparsers.add_parser("compare", help="Create outputs/comparison_summary.csv from finished runs.")
    compare_parser.add_argument("--outputs-dir", default="outputs")

    opt_parser = subparsers.add_parser("optional-pretrained-attention", help="Question 2.2(c): attention for torchvision vit_b_16.")
    opt_parser.add_argument("--config", default="config/config.yaml")
    opt_parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12])
    opt_parser.add_argument("--target-class", type=int, default=None)
    opt_parser.add_argument("--index", type=int, default=None)

    return parser


def main() -> None:
    """Dispatch CLI commands."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_from_args(args)
    elif args.command == "train-many":
        run_cases(args.config, args.cases, quick=args.quick, data_dir=args.data_dir)
    elif args.command == "evaluate":
        evaluate_from_args(args)
    elif args.command == "attention":
        attention_from_args(args)
    elif args.command == "compare":
        compare_results(args.outputs_dir)
    elif args.command == "optional-pretrained-attention":
        pretrained_vit_attention(args.config, args.layers, args.target_class, args.index)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
