"""
OCR Extractor
Uses PaddleOCR to extract text from P&ID images.
Falls back to pytesseract or metadata extraction gracefully.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field


@dataclass
class TextRegion:
    text: str
    confidence: float
    bbox: list[int]          # [x1, y1, x2, y2]
    is_tag: bool = False
    tag_type: str = ""       # e.g. "instrument", "equipment", "stream"


TAG_PREFIXES = {
    "T-": "equipment", "P-": "equipment", "E-": "equipment",
    "V-": "equipment", "R-": "equipment", "C-": "equipment",
    "HV-": "valve", "CV-": "valve", "FCV-": "valve", "LCV-": "valve",
    "PCV-": "valve", "TCV-": "valve",
    "FIC": "instrument", "PIC": "instrument", "TIC": "instrument",
    "LIC": "instrument", "FI-": "instrument", "PI-": "instrument",
    "TI-": "instrument", "LI-": "instrument", "FIT": "instrument",
    "L-": "stream", "LS-": "stream",
}


def _classify_tag(text: str) -> tuple[bool, str]:
    t = text.strip().upper()
    for prefix, ttype in TAG_PREFIXES.items():
        if t.startswith(prefix):
            return True, ttype
    return False, ""


class OCRExtractor:
    """
    Extract text from P&ID images.

    Priority: PaddleOCR → pytesseract → metadata sidecar.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg     = cfg or {}
        ocr_cfg      = self.cfg.get("ocr", {})
        self.lang    = ocr_cfg.get("lang", "en")
        self.use_gpu = ocr_cfg.get("use_gpu", False)
        self.min_conf = ocr_cfg.get("min_confidence", 0.4)
        self._paddle = None
        self._tess   = False
        self._init_backend()

    def _init_backend(self):
        try:
            from paddleocr import PaddleOCR
            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            print("[OCR] Using PaddleOCR backend")
        except Exception:
            try:
                import pytesseract
                self._tess = True
                print("[OCR] Using pytesseract backend")
            except Exception:
                print("[OCR] No OCR backend available — metadata fallback only")

    def extract(self, image: Image.Image,
                metadata_path: str | Path | None = None,
                metadata: dict | None = None) -> list[TextRegion]:
        if self._paddle:
            regions = self._run_paddle(image)
        elif self._tess:
            regions = self._run_tess(image)
        else:
            regions = []

        if not regions and metadata:
            regions = self._from_metadata_dict(metadata)
        elif not regions and metadata_path:
            regions = self._from_metadata(Path(metadata_path))

        # Post-process: classify tags
        for r in regions:
            r.is_tag, r.tag_type = _classify_tag(r.text)

        return regions

    def extract_file(self, image_path: str | Path) -> list[TextRegion]:
        p = Path(image_path)
        img = Image.open(p).convert("RGB")
        sidecar = p.with_suffix(".json")
        return self.extract(img, metadata_path=sidecar if sidecar.exists() else None)

    # ── PaddleOCR ─────────────────────────────────────────────────────────
    def _run_paddle(self, image: Image.Image) -> list[TextRegion]:
        arr = np.array(image)
        try:
            result = self._paddle.ocr(arr, cls=True)
        except Exception as e:
            print(f"[OCR] PaddleOCR error: {e}")
            return []
        if not result or not result[0]:
            return []
        regions = []
        for line in result[0]:
            if line is None:
                continue
            box_pts, (text, conf) = line
            if conf < self.min_conf:
                continue
            xs = [p[0] for p in box_pts]
            ys = [p[1] for p in box_pts]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            regions.append(TextRegion(text=text.strip(), confidence=float(conf), bbox=bbox))
        return regions

    # ── pytesseract ───────────────────────────────────────────────────────
    def _run_tess(self, image: Image.Image) -> list[TextRegion]:
        import pytesseract
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        regions = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = int(data["conf"][i])
            if conf < int(self.min_conf * 100):
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            regions.append(TextRegion(
                text=text, confidence=conf / 100.0,
                bbox=[x, y, x + w, y + h],
            ))
        return regions

    # ── Metadata fallback ─────────────────────────────────────────────────
    def _from_metadata_dict(self, meta: dict) -> list[TextRegion]:
        regions = []
        for sym in meta.get("symbols", []):
            tag = sym.get("tag", "")
            desc = sym.get("description", "")
            bbox = sym.get("bbox_pixels", [0, 0, 10, 10])
            if tag:
                regions.append(TextRegion(text=tag, confidence=1.0, bbox=bbox))
            if desc and desc != tag:
                lb = [bbox[0], bbox[3] + 5, bbox[2], bbox[3] + 30]
                regions.append(TextRegion(text=desc, confidence=0.95, bbox=lb))
        for conn in meta.get("connections", []):
            stream = conn.get("stream", "")
            if stream:
                regions.append(TextRegion(text=stream, confidence=0.9, bbox=[0, 0, 60, 20]))
        return regions

    def _from_metadata(self, meta_path: Path) -> list[TextRegion]:
        with open(meta_path) as f:
            meta = json.load(f)
        regions = []
        for sym in meta.get("symbols", []):
            tag = sym.get("tag", "")
            desc = sym.get("description", "")
            bbox = sym.get("bbox_pixels", [0, 0, 10, 10])
            if tag:
                regions.append(TextRegion(text=tag, confidence=1.0, bbox=bbox))
            if desc and desc != tag:
                # Place label below bbox
                lb = [bbox[0], bbox[3] + 5, bbox[2], bbox[3] + 30]
                regions.append(TextRegion(text=desc, confidence=0.95, bbox=lb))
        # Stream labels from connections
        for conn in meta.get("connections", []):
            stream = conn.get("stream", "")
            if stream:
                regions.append(TextRegion(text=stream, confidence=0.9,
                                          bbox=[0, 0, 60, 20]))
        return regions

    # ── Annotate helper ───────────────────────────────────────────────────
    def annotate(self, image: Image.Image,
                 regions: list[TextRegion]) -> Image.Image:
        """Return image with text region boxes drawn."""
        import cv2
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            color = (0, 160, 0) if r.is_tag else (160, 100, 0)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img_np, r.text[:20], (x1, max(y1 - 3, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
