#!/usr/bin/env python
"""
Train YOLOv8-nano on the synthetic P&ID dataset.

Usage
-----
# 1. Generate dataset first
python scripts/generate_dataset.py --n 200

# 2. Train
python scripts/train_model.py --epochs 50 --batch 16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.config import load_config
from src.detection.trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="Train YOLO P&ID detector")
    p.add_argument("--data",   default="data/yolo_dataset/dataset.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--model",  default="yolov8n.pt", help="Pre-trained backbone")
    args = p.parse_args()

    cfg = load_config()
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["batch"]  = args.batch

    trainer = Trainer(cfg)
    model_path = trainer.train(dataset_yaml=args.data, pretrained=args.model)
    print(f"\nModel saved to: {model_path}")
    print("Next step: streamlit run app/app.py")


if __name__ == "__main__":
    main()
