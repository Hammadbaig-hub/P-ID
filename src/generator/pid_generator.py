"""
P&ID Image Generator
Generates realistic synthetic P&ID diagrams with ISA S5.1 compliant symbols.
Produces PIL images alongside JSON metadata for YOLO training.
"""

import io
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from typing import Optional

from .symbols import (
    draw_gate_valve, draw_check_valve, draw_control_valve, draw_ball_valve,
    draw_centrifugal_pump, draw_tank, draw_vessel, draw_heat_exchanger,
    draw_instrument, draw_pipe, draw_title_block,
)

# ── Figure / coordinate constants ──────────────────────────────────────────
FIG_W = 24.0     # inches
FIG_H = 12.0     # inches
DPI   = 100
IMG_W = int(FIG_W * DPI)   # 2400
IMG_H = int(FIG_H * DPI)   # 1200

# Main process pipe height
PY = 6.5

# Class IDs (must match config.yaml)
CLS = {
    "gate_valve": 0, "globe_valve": 1, "check_valve": 2,
    "control_valve": 3, "ball_valve": 4, "butterfly_valve": 5,
    "centrifugal_pump": 6, "tank": 7, "vessel": 8,
    "column": 9, "heat_exchanger": 10,
    "pressure_instrument": 11, "temperature_instrument": 12,
    "flow_instrument": 13, "level_instrument": 14, "analyzer_instrument": 15,
}


def _bbox_to_pixel(x1: float, y1: float, x2: float, y2: float):
    """Convert data-coordinate bbox to pixel bbox (top-left origin)."""
    px1 = int(x1 * DPI)
    py1 = int((FIG_H - y2) * DPI)
    px2 = int(x2 * DPI)
    py2 = int((FIG_H - y1) * DPI)
    return (
        max(0, px1), max(0, py1),
        min(IMG_W, px2), min(IMG_H, py2),
    )


def _bbox_to_yolo(x1, y1, x2, y2):
    """Convert data-coordinate bbox to YOLO format (cx, cy, w, h) normalised."""
    px1, py1, px2, py2 = _bbox_to_pixel(x1, y1, x2, y2)
    cx = (px1 + px2) / 2 / IMG_W
    cy = (py1 + py2) / 2 / IMG_H
    bw = (px2 - px1) / IMG_W
    bh = (py2 - py1) / IMG_H
    return round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)


def _setup_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img.convert("RGB")


class PIDGenerator:
    """
    Generates synthetic P&ID diagrams.

    Usage
    -----
    gen = PIDGenerator()
    img, meta = gen.generate(seed=42)
    img.save("pid.png")
    """

    def generate(self, seed: Optional[int] = None) -> tuple[Image.Image, dict]:
        """Return (PIL Image, metadata dict) for one P&ID diagram."""
        if seed is not None:
            np.random.seed(seed)

        fig, ax = _setup_figure()
        symbols: list[dict] = []
        connections: list[dict] = []

        def rec(class_name, tag, description, bbox_data, connections_list=None):
            x1, y1, x2, y2 = bbox_data
            px_bbox = _bbox_to_pixel(x1, y1, x2, y2)
            yolo   = _bbox_to_yolo(x1, y1, x2, y2)
            symbols.append({
                "class_name": class_name,
                "class_id": CLS[class_name],
                "tag": tag,
                "description": description,
                "bbox_pixels": list(px_bbox),
                "bbox_norm": list(yolo),
                "connections": connections_list or [],
            })

        # ── Background unit boxes ───────────────────────────────────────────
        units = [
            (0.3, 3.5, 5.2, 11.0, "#f0f4ff", "FEED UNIT"),
            (5.3, 3.5, 13.5, 11.0, "#fff8f0", "REACTION UNIT"),
            (13.6, 3.5, 24.0 - 0.3, 11.0, "#f0fff4", "SEPARATION & STORAGE"),
        ]
        for ux1, uy1, ux2, uy2, uc, ulbl in units:
            ax.add_patch(mpatches.FancyBboxPatch(
                (ux1, uy1), ux2 - ux1, uy2 - uy1,
                boxstyle="round,pad=0.1", fc=uc, ec="#cccccc", lw=0.8,
                alpha=0.45, zorder=0,
            ))
            ax.text((ux1 + ux2) / 2, uy2 - 0.35, ulbl,
                    ha="center", va="top", fontsize=7, color="#555555",
                    zorder=1, fontstyle="italic")

        # ── Equipment ──────────────────────────────────────────────────────
        # T-101  Feed Storage Tank
        t101_x, t101_y = 2.0, PY
        bbox, in_pt, out_pt = draw_tank(ax, t101_x, t101_y, w=1.7, h=4.6, tag="T-101")
        rec("tank", "T-101", "Feed Storage Tank", bbox, ["P-101A", "P-101B"])
        t101_out = out_pt   # right nozzle

        # T-102  Product Storage Tank
        t102_x, t102_y = 22.0, PY
        bbox, in_pt, _ = draw_tank(ax, t102_x, t102_y, w=1.6, h=4.2, tag="T-102")
        rec("tank", "T-102", "Product Storage Tank", bbox, [])
        t102_in = in_pt     # left nozzle

        # P-101A  Feed Pump A (duty)
        p101a_x, p101a_y = 5.2, PY
        bbox, suc, dis = draw_centrifugal_pump(ax, p101a_x, p101a_y, r=0.48, tag="P-101A")
        rec("centrifugal_pump", "P-101A", "Feed Pump A (Duty)", bbox, ["E-101"])
        p101a_suc = suc
        p101a_dis = dis

        # P-101B  Feed Pump B (standby)  — below main line
        p101b_x, p101b_y = 5.2, PY - 2.5
        bbox, _, _ = draw_centrifugal_pump(ax, p101b_x, p101b_y, r=0.45, tag="P-101B")
        rec("centrifugal_pump", "P-101B", "Feed Pump B (Standby)", bbox, [])

        # E-101  Feed Pre-heater
        e101_x, e101_y = 9.5, PY
        bbox, he_in, he_out = draw_heat_exchanger(ax, e101_x, e101_y, w=2.8, h=1.1, tag="E-101")
        rec("heat_exchanger", "E-101", "Feed Pre-heater", bbox, ["R-101"])
        e101_in  = he_in
        e101_out = he_out

        # R-101  Reactor
        r101_x, r101_y = 13.0, PY
        bbox, v_in, v_out = draw_vessel(ax, r101_x, r101_y, w=1.3, h=4.8, tag="R-101")
        rec("vessel", "R-101", "Reactor", bbox, ["E-102"])
        r101_in  = v_in
        r101_out = v_out

        # E-102  Product Cooler
        e102_x, e102_y = 17.2, PY
        bbox, c_in, c_out = draw_heat_exchanger(ax, e102_x, e102_y, w=2.5, h=0.95, tag="E-102")
        rec("heat_exchanger", "E-102", "Product Cooler", bbox, ["V-101"])
        e102_in  = c_in
        e102_out = c_out

        # V-101  Flash Separator
        v101_x, v101_y = 20.5, PY
        bbox, sep_in, _ = draw_vessel(ax, v101_x, v101_y, w=1.2, h=3.8, tag="V-101")
        rec("vessel", "V-101", "Flash Separator", bbox, ["T-102"])
        v101_in = sep_in
        v101_out = (v101_x + 0.6, v101_y)

        # ── Inline Valves ─────────────────────────────────────────────────
        # HV-101  Tank outlet isolation (gate)
        hv101_x = 3.4
        bbox, hv101_in, hv101_out = draw_gate_valve(ax, hv101_x, PY, size=0.50)
        rec("gate_valve", "HV-101", "Feed Tank Outlet Isolation", bbox, ["P-101A"])
        ax.text(hv101_x, PY - 0.45, "HV-101", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # CV-101  Check valve after pump
        cv101_x = 6.7
        bbox, cv101_in, cv101_out = draw_check_valve(ax, cv101_x, PY, size=0.45)
        rec("check_valve", "CV-101", "Pump Discharge Check Valve", bbox)
        ax.text(cv101_x, PY - 0.42, "CV-101", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # FCV-101  Feed Flow Control Valve
        fcv101_x = 8.0
        bbox, fcv101_in, fcv101_out = draw_control_valve(ax, fcv101_x, PY, size=0.48, fail="FC")
        rec("control_valve", "FCV-101", "Feed Flow Control Valve", bbox, ["E-101"])
        ax.text(fcv101_x, PY - 0.42, "FCV-101", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # HV-102  HX to Reactor isolation
        hv102_x = 11.3
        bbox, hv102_in, hv102_out = draw_gate_valve(ax, hv102_x, PY, size=0.50)
        rec("gate_valve", "HV-102", "HX-Reactor Isolation", bbox)
        ax.text(hv102_x, PY - 0.45, "HV-102", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # HV-103  Reactor outlet isolation
        hv103_x = 14.7
        bbox, hv103_in, hv103_out = draw_gate_valve(ax, hv103_x, PY, size=0.50)
        rec("gate_valve", "HV-103", "Reactor Outlet Isolation", bbox)
        ax.text(hv103_x, PY - 0.45, "HV-103", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # LCV-101  Separator Level Control Valve
        lcv101_x = 21.5
        bbox, lcv101_in, lcv101_out = draw_control_valve(ax, lcv101_x, PY, size=0.48, fail="FO")
        rec("control_valve", "LCV-101", "Separator Level Control Valve", bbox, ["T-102"])
        ax.text(lcv101_x, PY - 0.42, "LCV-101", ha="center", va="top",
                fontsize=5.5, color="#1a1a1a", zorder=6, fontfamily="monospace")

        # HV-104  Product tank inlet isolation
        hv104_x = 22.0 - 0.75 - 0.32 - 0.15  # just before T-102 left nozzle
        # place between lcv101_out and t102_in
        hv104_x = 21.0 + (t102_in[0] - 21.5 - 0.48 * 0.8) * 0.5
        hv104_x = (lcv101_out[0] + t102_in[0]) / 2
        if hv104_x - 0.4 < lcv101_out[0]:
            hv104_x = lcv101_out[0] + 0.55
        bbox, _, _ = draw_gate_valve(ax, hv104_x, PY, size=0.45)
        rec("gate_valve", "HV-104", "Product Tank Inlet Isolation", bbox)

        # ── Main Pipe Segments ────────────────────────────────────────────
        segments = [
            # T-101 out → HV-101 in
            [t101_out, (hv101_in[0], PY)],
            # HV-101 out → P-101A suction
            [(hv101_out[0], PY), p101a_suc],
            # P-101A discharge → CV-101 in
            [p101a_dis, (cv101_in[0], PY)],
            # CV-101 out → FCV-101 in
            [(cv101_out[0], PY), (fcv101_in[0], PY)],
            # FCV-101 out → E-101 in
            [(fcv101_out[0], PY), e101_in],
            # E-101 out → HV-102 in
            [e101_out, (hv102_in[0], PY)],
            # HV-102 out → R-101 in
            [(hv102_out[0], PY), r101_in],
            # R-101 out → HV-103 in
            [r101_out, (hv103_in[0], PY)],
            # HV-103 out → E-102 in
            [(hv103_out[0], PY), e102_in],
            # E-102 out → V-101 in
            [e102_out, v101_in],
            # V-101 out → LCV-101 in
            [v101_out, (lcv101_in[0], PY)],
            # LCV-101 out → T-102 in
            [(lcv101_out[0], PY), t102_in],
        ]
        for pts in segments:
            draw_pipe(ax, pts, lw=2.2)

        # ── P-101B parallel branch ────────────────────────────────────────
        branch_x = (hv101_out[0] + p101a_suc[0]) / 2
        rejoin_x = p101a_dis[0]
        # Drop to P-101B level
        draw_pipe(ax, [(branch_x, PY), (branch_x, p101b_y)], lw=1.6)
        # Suction line to P-101B
        bbox_b, suc_b, dis_b = draw_centrifugal_pump.__wrapped__ if hasattr(draw_centrifugal_pump, "__wrapped__") else (None, None, None)
        draw_pipe(ax, [(branch_x, p101b_y), (p101b_x - 0.48 - 0.38 * 0.8, p101b_y)], lw=1.6)
        draw_pipe(ax, [(p101b_x + 0.48 + 0.4, p101b_y), (rejoin_x, p101b_y)], lw=1.6)
        draw_pipe(ax, [(rejoin_x, p101b_y), (rejoin_x, PY)], lw=1.6)
        # Gate valves on standby branch (simple marks)
        ax.text(branch_x, p101b_y - 0.25, "HV-105", ha="center", va="top",
                fontsize=5, color="#555555", zorder=6, fontstyle="italic")
        ax.text(rejoin_x - 0.1, p101b_y - 0.25, "HV-106", ha="center", va="top",
                fontsize=5, color="#555555", zorder=6, fontstyle="italic")

        # ── Instruments ───────────────────────────────────────────────────
        instr_above_y = 9.4
        instr_below_y = 3.8

        # FIC-101  Feed Flow Indicating Controller (below, connected to FCV-101)
        fic_x = fcv101_x
        fic_bbox = draw_instrument(ax, fic_x, instr_below_y, tag="FIC\n101",
                                   line_type="dotdash", connect_to=(fic_x, PY - 0.5))
        rec("flow_instrument", "FIC-101", "Feed Flow Indicating Controller", fic_bbox)
        ax.plot([fic_x, fic_x], [instr_below_y + 0.26, PY - 0.5], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)

        # PI-101  Pump Outlet Pressure Indicator (above, field-mounted)
        pi_x = (p101a_dis[0] + cv101_in[0]) / 2
        pi_bbox = draw_instrument(ax, pi_x, instr_above_y, tag="PI\n101",
                                  line_type="solid", connect_to=(pi_x, PY + 0.1))
        rec("pressure_instrument", "PI-101", "Pump Discharge Pressure Indicator", pi_bbox)
        ax.plot([pi_x, pi_x], [PY + 0.1, instr_above_y - 0.26], color="#1a1a1a", lw=0.85, zorder=4)

        # TIC-101  Preheater Outlet Temperature Controller (above, DCS)
        tic_x = e101_x + 0.3
        tic_bbox = draw_instrument(ax, tic_x, instr_above_y, tag="TIC\n101",
                                   line_type="dotdash", connect_to=(tic_x, PY + 0.1))
        rec("temperature_instrument", "TIC-101", "Feed Preheater Temp Controller", tic_bbox)
        ax.plot([tic_x, tic_x], [PY + 0.1, instr_above_y - 0.26], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)

        # PIC-101  Reactor Pressure Controller (above, DCS)
        pic_x = r101_x
        pic_bbox = draw_instrument(ax, pic_x, instr_above_y + 0.6, tag="PIC\n101",
                                   line_type="dotdash", connect_to=(r101_x + 0.65, PY + 0.8))
        rec("pressure_instrument", "PIC-101", "Reactor Pressure Controller", pic_bbox)
        ax.plot([r101_x + 0.65, r101_x + 0.65], [PY + 0.8, PY + 2.0], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)
        ax.plot([r101_x + 0.65, pic_x], [PY + 2.0, instr_above_y + 0.6 - 0.26], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)

        # TI-102  Reactor Temperature Indicator (below, field)
        ti_x = r101_x
        ti_bbox = draw_instrument(ax, ti_x, instr_below_y, tag="TI\n102",
                                  line_type="solid", connect_to=(r101_x + 0.65, PY - 0.8))
        rec("temperature_instrument", "TI-102", "Reactor Temperature Indicator", ti_bbox)
        ax.plot([r101_x + 0.65, r101_x + 0.65], [PY - 0.8, PY - 2.0], color="#1a1a1a", lw=0.85, zorder=4)
        ax.plot([r101_x + 0.65, ti_x], [PY - 2.0, instr_below_y + 0.26], color="#1a1a1a", lw=0.85, zorder=4)

        # TI-103  Cooler Outlet Temperature (below, field)
        ti103_x = (e102_out[0] + v101_in[0]) / 2
        ti103_bbox = draw_instrument(ax, ti103_x, instr_below_y, tag="TI\n103",
                                     line_type="solid", connect_to=(ti103_x, PY - 0.12))
        rec("temperature_instrument", "TI-103", "Cooler Outlet Temperature", ti103_bbox)
        ax.plot([ti103_x, ti103_x], [PY - 0.12, instr_below_y + 0.26], color="#1a1a1a", lw=0.85, zorder=4)

        # LIC-101  Separator Level Controller (below, DCS)
        lic_x = v101_x
        lic_bbox = draw_instrument(ax, lic_x, instr_below_y, tag="LIC\n101",
                                   line_type="dotdash", connect_to=(v101_x - 0.6, PY - 0.7))
        rec("level_instrument", "LIC-101", "Separator Level Controller", lic_bbox)
        ax.plot([v101_x - 0.6, v101_x - 0.6], [PY - 0.7, PY - 2.0], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)
        ax.plot([v101_x - 0.6, lic_x], [PY - 2.0, instr_below_y + 0.26], color="#1a1a1a", lw=0.85,
                linestyle="-.", zorder=4)

        # FI-102  Product Flow Indicator (above, field)
        fi102_x = (lcv101_out[0] + t102_in[0]) / 2
        fi102_bbox = draw_instrument(ax, fi102_x, instr_above_y, tag="FI\n102",
                                     line_type="solid", connect_to=(fi102_x, PY + 0.1))
        rec("flow_instrument", "FI-102", "Product Flow Indicator", fi102_bbox)
        ax.plot([fi102_x, fi102_x], [PY + 0.1, instr_above_y - 0.26], color="#1a1a1a", lw=0.85, zorder=4)

        # ── Connections metadata ──────────────────────────────────────────
        connections = [
            {"from": "T-101",   "to": "P-101A", "stream": "L-001", "fluid": "Raw Feed"},
            {"from": "T-101",   "to": "P-101B", "stream": "L-002", "fluid": "Raw Feed"},
            {"from": "P-101A",  "to": "E-101",  "stream": "L-003", "fluid": "Feed"},
            {"from": "E-101",   "to": "R-101",  "stream": "L-004", "fluid": "Hot Feed"},
            {"from": "R-101",   "to": "E-102",  "stream": "L-005", "fluid": "Reactor Effluent"},
            {"from": "E-102",   "to": "V-101",  "stream": "L-006", "fluid": "Cooled Product"},
            {"from": "V-101",   "to": "T-102",  "stream": "L-007", "fluid": "Liquid Product"},
        ]

        # ── Utility lines (steam, cooling water) ─────────────────────────
        # Steam to E-101 shell
        stm_x = e101_x - e101_x * 0.02 - 0.62
        ax.annotate("", xy=(stm_x, e101_y + 0.55 + 0.3), xytext=(stm_x, e101_y + 0.55 + 1.0),
                    arrowprops=dict(arrowstyle="->", color="#aa4400", lw=1.2), zorder=6)
        ax.text(stm_x, e101_y + 0.55 + 1.1, "STEAM", ha="center", va="bottom",
                fontsize=5.5, color="#aa4400", zorder=6, fontweight="bold")

        # CW to E-102 shell
        cw_x = e102_x + 0.62
        ax.annotate("", xy=(cw_x, e102_y + 0.48 + 0.3), xytext=(cw_x, e102_y + 0.48 + 0.95),
                    arrowprops=dict(arrowstyle="->", color="#0055aa", lw=1.2), zorder=6)
        ax.text(cw_x, e102_y + 0.48 + 1.05, "CW", ha="center", va="bottom",
                fontsize=5.5, color="#0055aa", zorder=6, fontweight="bold")

        # ── Stream numbers on pipe ────────────────────────────────────────
        stream_labels = [
            (3.4, PY + 0.18, "L-001"),
            ((p101a_dis[0] + cv101_in[0]) / 2, PY + 0.18, "L-003"),
            (e101_out[0] + 0.5, PY + 0.18, "L-004"),
            ((r101_out[0] + hv103_in[0]) / 2, PY + 0.18, "L-005"),
            ((e102_out[0] + v101_in[0]) / 2, PY + 0.18, "L-006"),
            ((lcv101_out[0] + t102_in[0]) / 2, PY + 0.18, "L-007"),
        ]
        for sx, sy, slbl in stream_labels:
            ax.text(sx, sy, slbl, ha="center", va="bottom", fontsize=5,
                    color="#444444", zorder=6, style="italic")

        # ── Grid (border) ─────────────────────────────────────────────────
        ax.add_patch(mpatches.Rectangle(
            (0.05, 0.05), FIG_W - 0.10, FIG_H - 0.10,
            fill=False, ec="#888888", lw=0.9, zorder=9,
        ))

        # ── Title block ───────────────────────────────────────────────────
        draw_title_block(
            ax,
            title="FEED PREPARATION TO PRODUCT RECOVERY",
            drawing_no="P-1001-001",
            revision="B1",
            date=datetime.date.today().strftime("%Y-%m-%d"),
        )

        # ── Render ────────────────────────────────────────────────────────
        img = _fig_to_pil(fig)
        plt.close(fig)

        metadata = {
            "image_size": [IMG_W, IMG_H],
            "drawing_no": "P-1001-001",
            "title": "FEED PREPARATION TO PRODUCT RECOVERY",
            "symbols": symbols,
            "connections": connections,
        }
        return img, metadata

    def generate_batch(self, n: int = 10, output_dir: str = "images/processed",
                       seed_start: int = 0) -> list[dict]:
        """Generate n P&ID images, save to disk, return list of metadata dicts."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results = []
        for i in range(n):
            seed = seed_start + i
            img, meta = self.generate(seed=seed)
            fname = f"pid_{seed:04d}.png"
            fpath = out / fname
            img.save(str(fpath))
            meta["image_path"] = str(fpath)
            json_path = out / f"pid_{seed:04d}.json"
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            results.append(meta)
        return results
