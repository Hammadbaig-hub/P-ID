"""
Dataset Generator
Creates YOLO-format training datasets from synthetic P&ID images.
"""

import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

from .pid_generator import PIDGenerator


def create_yolo_dataset(
    output_dir: str = "data/yolo_dataset",
    n_images: int = 200,
    seed: int = 0,
    split: dict | None = None,
) -> Path:
    """
    Generate n_images P&ID diagrams and organise them into a YOLO dataset.

    Dataset layout
    --------------
    output_dir/
        images/train/   val/   test/
        labels/train/   val/   test/
        dataset.yaml
    """
    split = split or {"train": 0.7, "val": 0.2, "test": 0.1}
    base = Path(output_dir)

    for subset in ["train", "val", "test"]:
        (base / "images" / subset).mkdir(parents=True, exist_ok=True)
        (base / "labels" / subset).mkdir(parents=True, exist_ok=True)

    gen = PIDGenerator()
    all_indices = list(range(n_images))
    random.seed(seed)
    random.shuffle(all_indices)

    n_train = int(n_images * split["train"])
    n_val   = int(n_images * split["val"])
    splits = {
        "train": all_indices[:n_train],
        "val":   all_indices[n_train: n_train + n_val],
        "test":  all_indices[n_train + n_val:],
    }

    print(f"Generating {n_images} P&ID images …")
    for subset, indices in splits.items():
        for idx in tqdm(indices, desc=subset):
            img, meta = gen.generate(seed=seed + idx)
            stem = f"pid_{seed + idx:05d}"

            # Save image
            img.save(str(base / "images" / subset / f"{stem}.png"))

            # Save YOLO annotation
            label_lines = []
            for sym in meta["symbols"]:
                cx, cy, bw, bh = sym["bbox_norm"]
                cid = sym["class_id"]
                label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_path = base / "labels" / subset / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines))

            # Save metadata sidecar
            (base / "images" / subset / f"{stem}.json").write_text(
                json.dumps(meta, indent=2)
            )

    # Write dataset.yaml
    class_names = [
        "gate_valve", "globe_valve", "check_valve", "control_valve",
        "ball_valve", "butterfly_valve", "centrifugal_pump", "tank", "vessel",
        "column", "heat_exchanger", "pressure_instrument",
        "temperature_instrument", "flow_instrument", "level_instrument",
        "analyzer_instrument",
    ]
    yaml_content = (
        f"path: {base.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(class_names)}\n"
        "names:\n" + "".join(f"  - {n}\n" for n in class_names)
    )
    (base / "dataset.yaml").write_text(yaml_content)
    print(f"Dataset written to {base.resolve()}")
    return base


class DatasetGenerator:
    """Thin wrapper for use from pipeline.py."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def create(self, n: int = 200, output_dir: str = "data/yolo_dataset") -> Path:
        return create_yolo_dataset(
            output_dir=output_dir,
            n_images=n,
            seed=self.cfg.get("training", {}).get("seed", 0),
            split=self.cfg.get("training", {}).get("split", None),
        )
