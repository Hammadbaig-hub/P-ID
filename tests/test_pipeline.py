"""Integration tests for the full pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PIL import Image
import networkx as nx


def test_pipeline_import():
    from src.pipeline import Pipeline
    assert Pipeline is not None


def test_pipeline_run_generated():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=0)

    assert result.image is not None
    assert isinstance(result.image, Image.Image)
    assert result.annotated_image is not None


def test_pipeline_detections():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=1)
    assert result.num_symbols >= 0
    # With metadata sidecar, we expect many detections
    assert result.num_symbols >= 5


def test_pipeline_ocr():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=2)
    assert result.num_text_regions >= 0


def test_pipeline_graph():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=3)
    assert result.graph is not None
    assert isinstance(result.graph, nx.DiGraph)
    assert result.num_graph_nodes >= 0


def test_pipeline_result_dicts():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=4)
    for d in result.detection_dicts:
        assert "tag" in d
        assert "class_name" in d
        assert "confidence" in d
        assert "bbox" in d


def test_pipeline_no_errors_on_generated():
    from src.pipeline import Pipeline
    pipe = Pipeline()
    result = pipe.run_generated(seed=5)
    # Only OCR / model-missing warnings expected, not crashes
    critical = [e for e in result.errors if "critical" in e.lower()]
    assert len(critical) == 0


def test_pipeline_chat_no_key():
    from src.pipeline import Pipeline
    pipe = Pipeline(api_key="")
    _ = pipe.run_generated(seed=6)
    resp = pipe.chat("What is T-101?")
    assert "API key" in resp or "key" in resp.lower() or "⚠" in resp


def test_graph_builder():
    from src.graph.builder import GraphBuilder
    from src.detection.detector import Detection

    dets = [
        Detection(7, "tank",            "T-101", 1.0, [100, 100, 300, 500], "Feed Tank"),
        Detection(6, "centrifugal_pump","P-101", 1.0, [400, 200, 600, 400], "Pump"),
        Detection(10,"heat_exchanger",  "E-101", 1.0, [700, 200, 1100,400], "Preheater"),
    ]
    builder = GraphBuilder()
    G = builder.build(dets, [], image_size=(2400, 1200))
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() >= 2


def test_graph_visualizer():
    from src.graph.builder import GraphBuilder
    from src.graph.visualizer import GraphVisualizer
    from src.detection.detector import Detection

    dets = [
        Detection(7, "tank",            "T-101", 1.0, [100, 100, 300, 500]),
        Detection(6, "centrifugal_pump","P-101", 1.0, [400, 200, 600, 400]),
    ]
    G = GraphBuilder().build(dets, [], image_size=(2400, 1200))
    viz = GraphVisualizer()
    fig = viz.plot(G)
    assert fig is not None
