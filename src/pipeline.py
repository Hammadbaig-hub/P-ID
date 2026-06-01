"""
P&ID Digitization Pipeline
Orchestrates: image generation → detection → OCR → graph → chatbot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
from PIL import Image

from .config import load_config, get_anthropic_key
from .generator.pid_generator import PIDGenerator
from .detection.detector import Detector, Detection
from .ocr.extractor import OCRExtractor, TextRegion
from .graph.builder import GraphBuilder
from .graph.visualizer import GraphVisualizer
from .chatbot.assistant import PIDAssistant


@dataclass
class PipelineResult:
    image: Image.Image
    annotated_image: Optional[Image.Image] = None
    detections: list[Detection] = field(default_factory=list)
    ocr_regions: list[TextRegion] = field(default_factory=list)
    graph: Optional[nx.DiGraph] = None
    graph_json: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # ── Convenience properties ─────────────────────────────────────────
    @property
    def detection_dicts(self) -> list[dict]:
        return [d.to_dict() for d in self.detections]

    @property
    def ocr_dicts(self) -> list[dict]:
        return [
            {
                "text": r.text,
                "confidence": round(r.confidence, 3),
                "bbox": r.bbox,
                "is_tag": r.is_tag,
                "tag_type": r.tag_type,
            }
            for r in self.ocr_regions
        ]

    @property
    def num_symbols(self) -> int:
        return len(self.detections)

    @property
    def num_text_regions(self) -> int:
        return len(self.ocr_regions)

    @property
    def num_graph_nodes(self) -> int:
        return self.graph.number_of_nodes() if self.graph else 0

    @property
    def num_graph_edges(self) -> int:
        return self.graph.number_of_edges() if self.graph else 0


class Pipeline:
    """
    End-to-end P&ID digitisation pipeline.

    Usage
    -----
    pipe = Pipeline()
    result = pipe.run_from_image(pil_image)
    result = pipe.run_generated(seed=42)
    """

    def __init__(self, cfg_path: str | None = None, api_key: str | None = None):
        self.cfg       = load_config(cfg_path)
        self._api_key  = api_key or get_anthropic_key()

        self.generator  = PIDGenerator()
        self.detector   = Detector(self.cfg)
        self.ocr        = OCRExtractor(self.cfg)
        self.graph_bld  = GraphBuilder(self.cfg)
        self.graph_viz  = GraphVisualizer(self.cfg)
        self.assistant  = PIDAssistant(self.cfg, api_key=self._api_key)

        self._last_result: Optional[PipelineResult] = None
        self._last_meta_path: Optional[Path] = None

    # ── Entry points ──────────────────────────────────────────────────────
    def run_generated(self, seed: int = 42,
                      save_dir: str | None = None) -> PipelineResult:
        """Generate a synthetic P&ID and run the full pipeline."""
        image, metadata = self.generator.generate(seed=seed)
        meta_path = None

        if save_dir:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            img_path  = out / f"pid_{seed:04d}.png"
            meta_path = out / f"pid_{seed:04d}.json"
            image.save(str(img_path))
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return self._run_pipeline(image, metadata=metadata, meta_path=meta_path)

    def run_from_image(self, image: Image.Image,
                       metadata_path: str | Path | None = None) -> PipelineResult:
        """Run the full pipeline on a supplied PIL Image."""
        meta = {}
        if metadata_path:
            with open(metadata_path) as f:
                meta = json.load(f)
        return self._run_pipeline(image, metadata=meta,
                                  meta_path=Path(metadata_path) if metadata_path else None)

    def run_from_file(self, image_path: str | Path) -> PipelineResult:
        """Run from an image file; auto-discovers sidecar JSON."""
        p = Path(image_path)
        image = Image.open(p).convert("RGB")
        sidecar = p.with_suffix(".json")
        return self.run_from_image(image,
                                   metadata_path=sidecar if sidecar.exists() else None)

    # ── Core pipeline ─────────────────────────────────────────────────────
    def _run_pipeline(self, image: Image.Image,
                      metadata: dict | None = None,
                      meta_path: Path | None = None) -> PipelineResult:
        result = PipelineResult(image=image, metadata=metadata or {})
        img_size = (image.width, image.height)

        # 1 ── Detection
        try:
            result.detections = self.detector.detect(
                image,
                metadata_path=meta_path,
                metadata=metadata if metadata and metadata.get("symbols") else None,
            )
            result.annotated_image = self.detector.annotate(image, result.detections)
        except Exception as e:
            result.errors.append(f"Detection error: {e}")
            result.annotated_image = image.copy()

        # 2 ── OCR
        try:
            result.ocr_regions = self.ocr.extract(
                image,
                metadata_path=meta_path,
                metadata=metadata if metadata and metadata.get("symbols") else None,
            )
        except Exception as e:
            result.errors.append(f"OCR error: {e}")

        # 3 ── Graph
        try:
            result.graph = self.graph_bld.build(
                result.detections,
                result.ocr_regions,
                metadata_path=meta_path,
                image_size=img_size,
            )
            result.graph_json = self.graph_bld.to_json(result.graph)
        except Exception as e:
            result.errors.append(f"Graph error: {e}")
            result.graph = nx.DiGraph()
            result.graph_json = {}

        # 4 ── Load chatbot context
        try:
            self.assistant.load_context(result.graph, result.ocr_regions)
        except Exception as e:
            result.errors.append(f"Chatbot context error: {e}")

        self._last_result = result
        self._last_meta_path = meta_path
        return result

    # ── Chat proxy ────────────────────────────────────────────────────────
    def chat(self, message: str) -> str:
        return self.assistant.chat(message)

    def chat_stream(self, message: str):
        yield from self.assistant.chat_stream(message)

    def update_api_key(self, key: str):
        """Hot-swap the Anthropic API key."""
        self._api_key = key
        self.assistant.client = None
        if key:
            import anthropic as _ant
            self.assistant.client = _ant.Anthropic(api_key=key)
