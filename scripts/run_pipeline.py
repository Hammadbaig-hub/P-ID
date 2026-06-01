#!/usr/bin/env python
"""
Run the full P&ID digitisation pipeline on an image (or generate one).

Usage
-----
# Generate a sample and run the pipeline:
python scripts/run_pipeline.py --generate --seed 42

# Run on an existing image:
python scripts/run_pipeline.py --image images/raw/my_pid.png

# Save outputs:
python scripts/run_pipeline.py --generate --seed 42 --out data/exports/run1
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline


def main():
    p = argparse.ArgumentParser(description="P&ID Pipeline CLI")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--generate", action="store_true", help="Generate a synthetic P&ID")
    grp.add_argument("--image",    type=str,           help="Path to input image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  type=str, default="data/exports/run",
                   help="Output directory for results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline()

    print("Running pipeline …")
    if args.generate:
        result = pipe.run_generated(seed=args.seed, save_dir=str(out / "images"))
    else:
        result = pipe.run_from_file(args.image)

    # Save annotated image
    if result.annotated_image:
        result.annotated_image.save(str(out / "annotated.png"))
        print(f"  Annotated image → {out / 'annotated.png'}")

    # Save detections JSON
    det_path = out / "detections.json"
    det_path.write_text(json.dumps(result.detection_dicts, indent=2))
    print(f"  Detections ({result.num_symbols}) → {det_path}")

    # Save OCR JSON
    ocr_path = out / "ocr.json"
    ocr_path.write_text(json.dumps(result.ocr_dicts, indent=2))
    print(f"  OCR regions ({result.num_text_regions}) → {ocr_path}")

    # Save graph
    if result.graph_json:
        gj_path = out / "graph.json"
        gj_path.write_text(json.dumps(result.graph_json, indent=2))
        print(f"  Graph ({result.num_graph_nodes} nodes, {result.num_graph_edges} edges) → {gj_path}")

    if result.errors:
        print("\nWarnings:")
        for e in result.errors:
            print(f"  ⚠ {e}")

    print(f"\nDone. Open {out} for outputs.")


if __name__ == "__main__":
    main()
