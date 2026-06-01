#!/usr/bin/env python
"""
Generate a YOLO-format training dataset of synthetic P&ID images.

Usage
-----
python scripts/generate_dataset.py --n 200 --out data/yolo_dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.generator.dataset_generator import create_yolo_dataset


def main():
    p = argparse.ArgumentParser(description="Generate YOLO P&ID dataset")
    p.add_argument("--n",   type=int, default=200, help="Number of images")
    p.add_argument("--out", type=str, default="data/yolo_dataset", help="Output directory")
    p.add_argument("--seed",type=int, default=0,   help="Random seed")
    args = p.parse_args()

    out = create_yolo_dataset(
        output_dir=args.out,
        n_images=args.n,
        seed=args.seed,
    )
    print(f"\nDataset ready at: {out.resolve()}")
    print("Next step:  python scripts/train_model.py")


if __name__ == "__main__":
    main()
