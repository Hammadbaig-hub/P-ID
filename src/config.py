import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CFG_PATH = _ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else _CFG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(_ROOT)
    return cfg


def get_class_names(cfg: dict) -> list[str]:
    return [s["name"] for s in cfg["symbols"]["classes"]]


def get_class_map(cfg: dict) -> dict[str, int]:
    return {s["name"]: s["id"] for s in cfg["symbols"]["classes"]}


def get_anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return key
