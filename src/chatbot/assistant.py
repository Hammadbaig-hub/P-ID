"""
P&ID AI Assistant
Anthropic-powered chatbot that reasons over the extracted P&ID knowledge graph.
"""

import os
from pathlib import Path
from typing import Generator

import anthropic
import networkx as nx


_SYSTEM_TEMPLATE = Path(__file__).parent.parent.parent / "chatbot" / "prompts" / "system_prompt.txt"


def _load_system_template() -> str:
    try:
        return _SYSTEM_TEMPLATE.read_text()
    except FileNotFoundError:
        return (
            "You are an expert P&ID analyst. The following data has been extracted:\n\n"
            "Equipment: {equipment_list}\nConnections: {connections_list}\n"
            "Instruments: {instruments_list}\nOCR Text: {ocr_text}\n\n"
            "Answer engineering questions about this P&ID accurately."
        )


def _graph_to_context(G: nx.DiGraph, ocr_regions: list) -> dict:
    """Serialise graph + OCR into template substitution dict."""
    equipment_lines, instrument_lines, connection_lines = [], [], []

    INSTRUMENT_CLASSES = {
        "pressure_instrument", "temperature_instrument",
        "flow_instrument", "level_instrument", "analyzer_instrument",
    }

    for node, data in G.nodes(data=True):
        cls  = data.get("class_name", "unknown")
        desc = data.get("description", "")
        line = f"  • {node}: {desc} ({cls})"
        if cls in INSTRUMENT_CLASSES:
            instrument_lines.append(line)
        else:
            equipment_lines.append(line)

    for u, v, data in G.edges(data=True):
        stream = data.get("stream", "")
        fluid  = data.get("fluid", "")
        connection_lines.append(f"  • {u} → {v}  [{stream}] {fluid}")

    ocr_texts = [r.text for r in ocr_regions if r.text.strip()]
    return {
        "equipment_list":  "\n".join(equipment_lines) or "  (no equipment detected)",
        "connections_list": "\n".join(connection_lines) or "  (no connections detected)",
        "instruments_list": "\n".join(instrument_lines) or "  (no instruments detected)",
        "ocr_text":         "  " + ", ".join(ocr_texts[:60]) if ocr_texts else "  (no text extracted)",
    }


class PIDAssistant:
    """
    Conversational assistant for P&ID analysis.

    Parameters
    ----------
    cfg : dict from config.yaml
    api_key : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    """

    def __init__(self, cfg: dict | None = None, api_key: str | None = None):
        self.cfg     = cfg or {}
        bot_cfg      = self.cfg.get("chatbot", {})
        self.model   = bot_cfg.get("model", "claude-opus-4-8")
        self.max_tok = bot_cfg.get("max_tokens", 2048)
        self.temp    = bot_cfg.get("temperature", 0.3)

        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = anthropic.Anthropic(api_key=key) if key else None

        self._system_tmpl = _load_system_template()
        self._system_prompt = "P&ID analyst assistant. No diagram loaded yet."
        self.history: list[dict] = []

    # ── Context loading ───────────────────────────────────────────────────
    def load_context(self, G: nx.DiGraph, ocr_regions: list):
        ctx = _graph_to_context(G, ocr_regions)
        self._system_prompt = self._system_tmpl.format(**ctx)
        self.history = []

    # ── Synchronous chat (full response) ─────────────────────────────────
    def chat(self, user_message: str) -> str:
        if not self.client:
            return (
                "⚠️ Anthropic API key not configured. "
                "Set the ANTHROPIC_API_KEY environment variable and restart."
            )
        self.history.append({"role": "user", "content": user_message})
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tok,
                temperature=self.temp,
                system=self._system_prompt,
                messages=self.history,
            )
            text = resp.content[0].text
        except anthropic.AuthenticationError:
            return "❌ Invalid Anthropic API key."
        except anthropic.RateLimitError:
            return "⏳ Rate limit hit. Please wait a moment and try again."
        except Exception as e:
            return f"❌ API error: {e}"

        self.history.append({"role": "assistant", "content": text})
        return text

    # ── Streaming chat (yields text chunks) ──────────────────────────────
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        if not self.client:
            yield (
                "⚠️ Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY and restart."
            )
            return

        self.history.append({"role": "user", "content": user_message})
        full_text = ""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tok,
                temperature=self.temp,
                system=self._system_prompt,
                messages=self.history,
            ) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
                    yield chunk
        except anthropic.AuthenticationError:
            yield "❌ Invalid Anthropic API key."
            return
        except Exception as e:
            yield f"❌ API error: {e}"
            return

        self.history.append({"role": "assistant", "content": full_text})

    def clear_history(self):
        self.history = []

    # ── Suggested questions ───────────────────────────────────────────────
    def suggested_questions(self) -> list[str]:
        return [
            "What is the overall process flow from feed to product?",
            "List all control valves and their fail-safe positions.",
            "What instruments are monitoring the reactor R-101?",
            "Explain the purpose of each heat exchanger in this diagram.",
            "What are the safety interlocks visible in this P&ID?",
            "Which pump has a standby/spare and how is it configured?",
            "What utility streams (steam, cooling water) are used?",
            "Describe the level control strategy for the separator.",
        ]
