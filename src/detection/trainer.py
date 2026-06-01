"""
YOLO Trainer
Trains a YOLOv8-nano model on a generated P&ID dataset.
"""

from pathlib import Path


class Trainer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        t = self.cfg.get("training", {})
        self.epochs    = t.get("epochs", 50)
        self.batch     = t.get("batch", 16)
        self.img_size  = t.get("img_size", 640)
        self.patience  = t.get("patience", 10)
        self.model_out = Path(self.cfg.get("detection", {}).get(
            "model_path", "detection/models/pid_detector.pt"))
        self.model_out.parent.mkdir(parents=True, exist_ok=True)

    def train(self, dataset_yaml: str = "data/yolo_dataset/dataset.yaml",
              pretrained: str = "yolov8n.pt") -> Path:
        """Train YOLOv8-nano on the synthetic dataset. Returns path to best.pt."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        model = YOLO(pretrained)
        results = model.train(
            data=dataset_yaml,
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.img_size,
            patience=self.patience,
            project="detection/runs",
            name="pid_train",
            exist_ok=True,
            verbose=True,
        )

        # Copy best weights to canonical location
        best = Path("detection/runs/pid_train/weights/best.pt")
        if best.exists():
            import shutil
            shutil.copy(best, self.model_out)
            print(f"[Trainer] Model saved to {self.model_out}")

        return self.model_out
