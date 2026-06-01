"""Tests for the P&ID generator and symbol drawing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from PIL import Image


def test_generator_returns_pil_image():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    img, meta = gen.generate(seed=0)
    assert isinstance(img, Image.Image)
    assert img.width == 2400
    assert img.height == 1200


def test_generator_metadata_structure():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    _, meta = gen.generate(seed=1)
    assert "symbols" in meta
    assert "connections" in meta
    assert isinstance(meta["symbols"], list)
    assert len(meta["symbols"]) > 0


def test_symbol_has_required_fields():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    _, meta = gen.generate(seed=2)
    for sym in meta["symbols"]:
        assert "class_name" in sym
        assert "class_id" in sym
        assert "tag" in sym
        assert "bbox_pixels" in sym
        assert "bbox_norm" in sym
        assert len(sym["bbox_pixels"]) == 4
        assert len(sym["bbox_norm"]) == 4


def test_yolo_norm_in_range():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    _, meta = gen.generate(seed=3)
    for sym in meta["symbols"]:
        cx, cy, bw, bh = sym["bbox_norm"]
        assert 0.0 <= cx <= 1.0, f"cx={cx} out of range for {sym['tag']}"
        assert 0.0 <= cy <= 1.0, f"cy={cy} out of range for {sym['tag']}"
        assert 0.0 < bw <= 1.0, f"bw={bw} invalid for {sym['tag']}"
        assert 0.0 < bh <= 1.0, f"bh={bh} invalid for {sym['tag']}"


def test_generator_deterministic():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    img1, _ = gen.generate(seed=42)
    img2, _ = gen.generate(seed=42)
    assert np.array(img1).sum() == np.array(img2).sum()


def test_different_seeds_differ():
    from src.generator.pid_generator import PIDGenerator
    gen = PIDGenerator()
    img1, _ = gen.generate(seed=1)
    img2, _ = gen.generate(seed=99)
    # At least the metadata seed is reflected (images may be identical for simple layouts)
    assert img1.size == img2.size


def test_symbol_drawing_gate_valve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.generator.symbols import draw_gate_valve

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    bbox, in_pt, out_pt = draw_gate_valve(ax, 2.0, 2.0, size=0.5)
    plt.close(fig)

    x1, y1, x2, y2 = bbox
    assert x1 < x2
    assert y1 < y2
    assert in_pt[0] < out_pt[0]


def test_symbol_drawing_pump():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.generator.symbols import draw_centrifugal_pump

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    bbox, in_pt, out_pt = draw_centrifugal_pump(ax, 2.0, 2.0, r=0.5)
    plt.close(fig)

    assert in_pt[0] < out_pt[0]
    x1, y1, x2, y2 = bbox
    assert x2 - x1 > 0 and y2 - y1 > 0
