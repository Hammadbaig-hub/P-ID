# P&ID Digitization Pipeline

End-to-end system for converting scanned/photographed P&ID drawings into structured, queryable graph data with a conversational interface.

## Project Structure

```
p&id/
├── images/
│   ├── raw/            # Original scanned or photographed P&ID files
│   └── processed/      # Pre-processed images (denoised, deskewed, contrast-enhanced)
│
├── detection/
│   ├── models/         # Trained object detection model weights (YOLO, Detectron2, etc.)
│   ├── configs/        # Model configuration and training hyperparameters
│   └── outputs/        # Detection results — bounding boxes, class labels, confidence scores
│
├── ocr/
│   ├── models/         # OCR model weights or config (Tesseract, PaddleOCR, TrOCR, etc.)
│   └── outputs/        # Extracted text annotations linked to detected symbols/lines
│
├── graph/
│   ├── schemas/        # Graph schema definitions (node types, edge types, attributes)
│   └── outputs/        # Generated graph files (JSON-LD, GraphML, Neo4j exports)
│
├── chatbot/
│   ├── prompts/        # System prompts and few-shot examples for the LLM interface
│   └── memory/         # Conversation history and session state
│
├── data/
│   ├── annotations/    # Ground-truth labels for training and evaluation
│   └── exports/        # Final deliverables (JSON, CSV, reports)
│
├── notebooks/          # Exploratory analysis and prototyping (Jupyter)
├── scripts/            # Standalone utility and pipeline scripts
└── tests/              # Unit and integration tests
```

## Pipeline Overview

1. **Ingest** — Place raw P&ID images in `images/raw/`
2. **Pre-process** — Deskew, denoise, normalize → `images/processed/`
3. **Detect** — Run symbol/line detector → `detection/outputs/`
4. **OCR** — Extract tag numbers and labels → `ocr/outputs/`
5. **Build Graph** — Assemble equipment graph → `graph/outputs/`
6. **Query** — Ask questions via the chatbot interface

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline on a single image
python scripts/run_pipeline.py --input images/raw/diagram.png

# Launch the chatbot
python scripts/chatbot.py
```
