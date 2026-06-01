"""
Knowledge Graph Builder
Constructs a NetworkX DiGraph from detection + OCR results.
Nodes = equipment / instruments.  Edges = pipe connections.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
import networkx as nx

from ..detection.detector import Detection
from ..ocr.extractor import TextRegion


@dataclass
class NodeData:
    tag: str
    class_name: str
    description: str
    bbox: list[int]
    confidence: float
    properties: dict = field(default_factory=dict)


@dataclass
class EdgeData:
    stream: str
    fluid: str = ""
    properties: dict = field(default_factory=dict)


# Equipment that appears in the process main line (ordered)
PROCESS_ORDER = [
    "tank", "centrifugal_pump", "heat_exchanger", "vessel",
    "column", "heat_exchanger", "vessel",
]

INSTRUMENT_CLASSES = {
    "pressure_instrument", "temperature_instrument",
    "flow_instrument", "level_instrument", "analyzer_instrument",
}

VALVE_CLASSES = {
    "gate_valve", "globe_valve", "check_valve", "control_valve",
    "ball_valve", "butterfly_valve",
}


class GraphBuilder:
    """
    Build a knowledge graph from pipeline results.

    If metadata_path is supplied (generated images), use the ground-truth
    connection map.  Otherwise infer connections from spatial proximity.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def build(
        self,
        detections: list[Detection],
        ocr_regions: list[TextRegion],
        metadata_path: str | Path | None = None,
        image_size: tuple[int, int] = (2400, 1200),
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        # Map OCR tags to detections for enrichment
        tag_map = {r.text.strip().upper(): r for r in ocr_regions if r.is_tag}

        # Add nodes
        for det in detections:
            tag = det.tag or f"UNKNOWN-{det.class_id}"
            desc = det.description
            ocr_match = tag_map.get(tag.upper(), None)
            G.add_node(tag, **{
                "class_name": det.class_name,
                "description": desc,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "color": _node_color(det.class_name),
                "size": _node_size(det.class_name),
                "ocr_text": ocr_match.text if ocr_match else tag,
            })

        # Add edges
        if metadata_path:
            self._edges_from_metadata(G, Path(metadata_path))
        else:
            self._edges_from_proximity(G, detections, image_size)

        # Attach stream labels from OCR
        stream_texts = [r.text for r in ocr_regions if r.text.startswith("L-")]
        for i, (u, v) in enumerate(G.edges()):
            G[u][v].setdefault("stream", stream_texts[i] if i < len(stream_texts) else "")
            G[u][v].setdefault("fluid", "")

        return G

    # ── Edge building strategies ──────────────────────────────────────────
    def _edges_from_metadata(self, G: nx.DiGraph, meta_path: Path):
        with open(meta_path) as f:
            meta = json.load(f)
        for conn in meta.get("connections", []):
            src, dst = conn["from"], conn["to"]
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst,
                           stream=conn.get("stream", ""),
                           fluid=conn.get("fluid", ""))

        # Connect instruments to equipment
        sym_by_tag = {s["tag"]: s for s in meta.get("symbols", [])}
        for node_tag in list(G.nodes()):
            node_cls = G.nodes[node_tag].get("class_name", "")
            if node_cls in INSTRUMENT_CLASSES:
                # Heuristic: find the nearest non-instrument node
                node_bbox = G.nodes[node_tag].get("bbox", [0, 0, 0, 0])
                nc = _bbox_center(node_bbox)
                best, best_d = None, float("inf")
                for other in G.nodes():
                    if other == node_tag:
                        continue
                    other_cls = G.nodes[other].get("class_name", "")
                    if other_cls in INSTRUMENT_CLASSES:
                        continue
                    oc = _bbox_center(G.nodes[other].get("bbox", [0, 0, 0, 0]))
                    d = (nc[0] - oc[0]) ** 2 + (nc[1] - oc[1]) ** 2
                    if d < best_d:
                        best_d, best = d, other
                if best:
                    G.add_edge(node_tag, best, stream="signal", fluid="signal")

    def _edges_from_proximity(self, G: nx.DiGraph, detections: list[Detection],
                               image_size: tuple[int, int]):
        """Infer connections: sort non-instruments left-to-right and chain them."""
        W, H = image_size
        # Separate process equipment from instruments
        process = [d for d in detections
                   if d.class_name not in INSTRUMENT_CLASSES]
        instruments = [d for d in detections
                       if d.class_name in INSTRUMENT_CLASSES]

        # Sort by x centre
        process.sort(key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)

        stream_counter = 1
        for i in range(len(process) - 1):
            a, b = process[i], process[i + 1]
            if a.tag in G.nodes and b.tag in G.nodes:
                G.add_edge(a.tag, b.tag,
                           stream=f"L-{stream_counter:03d}", fluid="process")
                stream_counter += 1

        # Connect each instrument to the nearest process node
        for inst in instruments:
            if inst.tag not in G.nodes:
                continue
            ic = _bbox_center(inst.bbox)
            best, best_d = None, float("inf")
            for proc in process:
                if proc.tag not in G.nodes:
                    continue
                pc = _bbox_center(proc.bbox)
                d = (ic[0] - pc[0]) ** 2 + (ic[1] - pc[1]) ** 2
                if d < best_d:
                    best_d, best = d, proc.tag
            if best:
                G.add_edge(inst.tag, best, stream="signal", fluid="signal")

    # ── Export ────────────────────────────────────────────────────────────
    def to_json(self, G: nx.DiGraph) -> dict:
        return {
            "nodes": [
                {"id": n, **{k: v for k, v in d.items() if k != "bbox"}}
                for n, d in G.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in G.edges(data=True)
            ],
            "stats": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(G),
            },
        }

    def save_json(self, G: nx.DiGraph, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(G), f, indent=2)

    def save_graphml(self, G: nx.DiGraph, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # GraphML can't serialise list-type bbox; convert to string
        H = G.copy()
        for n in H.nodes():
            if "bbox" in H.nodes[n]:
                H.nodes[n]["bbox"] = str(H.nodes[n]["bbox"])
        nx.write_graphml(H, str(path))


# ── Helpers ───────────────────────────────────────────────────────────────

def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _node_color(class_name: str) -> str:
    colors = {
        "tank": "#4682B4",
        "vessel": "#5F9EA0",
        "column": "#6495ED",
        "centrifugal_pump": "#FF8C00",
        "heat_exchanger": "#DC143C",
        "gate_valve": "#2E8B57",
        "globe_valve": "#3CB371",
        "check_valve": "#20B2AA",
        "control_valve": "#9400D3",
        "ball_valve": "#008B8B",
        "butterfly_valve": "#006400",
        "pressure_instrument": "#DAA520",
        "temperature_instrument": "#CD5C5C",
        "flow_instrument": "#4169E1",
        "level_instrument": "#8B4513",
        "analyzer_instrument": "#B8860B",
    }
    return colors.get(class_name, "#888888")


def _node_size(class_name: str) -> int:
    large = {"tank", "vessel", "column", "heat_exchanger"}
    medium = {"centrifugal_pump"}
    if class_name in large:
        return 30
    if class_name in medium:
        return 22
    if class_name in INSTRUMENT_CLASSES:
        return 12
    return 16
