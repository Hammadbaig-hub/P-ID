"""
Symbol Detector
Wraps YOLO26 inference. Falls back to metadata-based mock detection when
no trained model is present (useful for generated images that have sidecar
JSON files).
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image

CLASS_NAMES = [
    "gate_valve", "globe_valve", "check_valve", "control_valve",
    "ball_valve", "butterfly_valve", "centrifugal_pump", "tank", "vessel",
    "column", "heat_exchanger", "pressure_instrument",
    "temperature_instrument", "flow_instrument", "level_instrument",
    "analyzer_instrument",
]

# Human-friendly labels
DISPLAY = {
    "gate_valve": "Gate Valve",
    "globe_valve": "Globe Valve",
    "check_valve": "Check Valve",
    "control_valve": "Control Valve",
    "ball_valve": "Ball Valve",
    "butterfly_valve": "Butterfly Valve",
    "centrifugal_pump": "Centrifugal Pump",
    "tank": "Storage Tank",
    "vessel": "Pressure Vessel",
    "column": "Column",
    "heat_exchanger": "Heat Exchanger",
    "pressure_instrument": "Pressure Instrument",
    "temperature_instrument": "Temperature Instrument",
    "flow_instrument": "Flow Instrument",
    "level_instrument": "Level Instrument",
    "analyzer_instrument": "Analyser Instrument",
}

# Colour per class (BGR for OpenCV, hex for display)
CLASS_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
]


class Detection:
    """Single symbol detection result."""
    __slots__ = ("class_id", "class_name", "tag", "confidence",
                 "bbox", "description")

    def __init__(self, class_id: int, class_name: str, tag: str,
                 confidence: float, bbox: list[int],
                 description: str = ""):
        self.class_id   = class_id
        self.class_name = class_name
        self.tag        = tag
        self.confidence = confidence
        self.bbox       = bbox          # [x1, y1, x2, y2] pixels
        self.description = description

    def to_dict(self) -> dict:
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "tag":        self.tag,
            "confidence": round(self.confidence, 4),
            "bbox":       self.bbox,
            "description": self.description,
            "color":      CLASS_COLORS[self.class_id % len(CLASS_COLORS)],
        }


class Detector:
    """
    Detect P&ID symbols in an image.

    Priority
    --------
    1. YOLO26 model (if model_path exists and ultralytics is installed)
    2. Metadata sidecar JSON (for generated images)
    3. Empty result with warning
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        det_cfg = self.cfg.get("detection", {})
        self.model_path = Path(det_cfg.get("model_path", "detection/models/pid_detector.pt"))
        self.conf       = det_cfg.get("confidence", 0.25)
        self.iou        = det_cfg.get("iou", 0.45)
        self.img_size   = det_cfg.get("img_size", 640)
        self.device     = det_cfg.get("device", "cpu")
        self._model     = None
        self._try_load_model()

    def _try_load_model(self):
        if not self.model_path.exists():
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            print(f"[Detector] Loaded YOLO26 model from {self.model_path}")
        except Exception as e:
            print(f"[Detector] Could not load model: {e}")

    def detect(self, image: Image.Image,
               metadata_path: str | Path | None = None,
               metadata: dict | None = None) -> list[Detection]:
        """Run detection on a PIL Image. Returns list of Detection objects."""
        if self._model is not None:
            return self._detect_yolo(image)
        if metadata:
            return self._detect_from_metadata_dict(metadata)
        if metadata_path:
            return self._detect_from_metadata(Path(metadata_path))
        return []

    def detect_file(self, image_path: str | Path) -> list[Detection]:
        """Detect from an image file, auto-discovering sidecar JSON."""
        img_path = Path(image_path)
        image = Image.open(img_path).convert("RGB")
        sidecar = img_path.with_suffix(".json")
        return self.detect(image, metadata_path=sidecar if sidecar.exists() else None)

    # ── YOLO inference ────────────────────────────────────────────────────
    def _detect_yolo(self, image: Image.Image) -> list[Detection]:
        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        for box in results.boxes:
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "unknown"
            detections.append(Detection(
                class_id=cid, class_name=cname,
                tag=f"{DISPLAY.get(cname, cname)[:3].upper()}-{len(detections)+101:03d}",
                confidence=conf, bbox=[x1, y1, x2, y2],
                description=DISPLAY.get(cname, cname),
            ))
        return detections

    # ── Metadata fallback ─────────────────────────────────────────────────
    def _detect_from_metadata_dict(self, meta: dict) -> list[Detection]:
        detections = []
        for sym in meta.get("symbols", []):
            cid   = sym["class_id"]
            cname = sym["class_name"]
            tag   = sym.get("tag", f"SYM-{len(detections)+1:03d}")
            bbox  = sym.get("bbox_pixels", [0, 0, 10, 10])
            detections.append(Detection(
                class_id=cid, class_name=cname, tag=tag,
                confidence=1.0, bbox=bbox,
                description=sym.get("description", DISPLAY.get(cname, cname)),
            ))
        return detections

    def _detect_from_metadata(self, meta_path: Path) -> list[Detection]:
        with open(meta_path) as f:
            meta = json.load(f)
        detections = []
        for sym in meta.get("symbols", []):
            cid   = sym["class_id"]
            cname = sym["class_name"]
            tag   = sym.get("tag", f"SYM-{len(detections)+1:03d}")
            bbox  = sym.get("bbox_pixels", [0, 0, 10, 10])
            detections.append(Detection(
                class_id=cid, class_name=cname, tag=tag,
                confidence=1.0, bbox=bbox,
                description=sym.get("description", DISPLAY.get(cname, cname)),
            ))
        return detections

    # ── Annotated image ───────────────────────────────────────────────────
    def annotate(self, image: Image.Image,
                 detections: list[Detection]) -> Image.Image:
        """Return a copy of image with coloured bounding boxes and labels."""
        import cv2
        img_np = np.array(image.copy())
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            hex_c = CLASS_COLORS[det.class_id % len(CLASS_COLORS)].lstrip("#")
            r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            color_bgr = (b, g, r)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, 2)
            label = f"{det.tag} [{det.confidence:.2f}]"
            fs, th = 0.45, 1
            (tw, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ly = max(y1 - 4, text_h + 4)
            cv2.rectangle(img_bgr, (x1, ly - text_h - 4), (x1 + tw + 4, ly), color_bgr, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
