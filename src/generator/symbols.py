"""
ISA S5.1 / ISO 10628 P&ID Symbol Drawing Library
All functions draw on a matplotlib Axes object using data coordinates.
Each function returns (x1, y1, x2, y2) bounding box and (in_pt, out_pt) connection points.
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, Ellipse, FancyArrow
from typing import Tuple

Color = str
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


def _rot(pts: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return (pts - [cx, cy]) @ R.T + [cx, cy]


def draw_pipe(ax, points: list, color: Color = "#1a1a1a", lw: float = 2.0, zorder: int = 1):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=zorder)


def draw_gate_valve(
    ax, cx: float, cy: float, size: float = 0.5,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Bowtie (two filled triangles tips touching) on a pipe."""
    h = size / 2
    s = size

    pts = np.array([
        [cx - s * 0.8, cy], [cx - h, cy],
        [cx + h, cy], [cx + s * 0.8, cy],
    ])
    ltri = np.array([[cx - h, cy + h], [cx - h, cy - h], [cx, cy]])
    rtri = np.array([[cx + h, cy + h], [cx + h, cy - h], [cx, cy]])

    if rotation:
        pts = _rot(pts, rotation, cx, cy)
        ltri = _rot(ltri, rotation, cx, cy)
        rtri = _rot(rtri, rotation, cx, cy)

    ax.plot(*pts[:2].T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot(*pts[2:].T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(Polygon(ltri, closed=True, fc=color, ec=color, lw=lw, zorder=3))
    ax.add_patch(Polygon(rtri, closed=True, fc=color, ec=color, lw=lw, zorder=3))

    in_pt = (pts[0, 0], pts[0, 1])
    out_pt = (pts[3, 0], pts[3, 1])
    return (cx - s * 0.8, cy - h, cx + s * 0.8, cy + h), in_pt, out_pt


def draw_globe_valve(
    ax, cx: float, cy: float, size: float = 0.5,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Circle with stem (globe valve)."""
    r = size / 2
    s = size

    stubs = np.array([
        [cx - s * 0.7, cy], [cx - r, cy],
        [cx + r, cy], [cx + s * 0.7, cy],
    ])
    stem = np.array([[cx, cy + r], [cx, cy + r * 1.5]])

    if rotation:
        stubs = _rot(stubs, rotation, cx, cy)
        stem = _rot(stem, rotation, cx, cy)
        circ_center = _rot(np.array([[cx, cy]]), rotation, cx, cy)[0]
    else:
        circ_center = np.array([cx, cy])

    ax.plot(*stubs[:2].T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot(*stubs[2:].T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(Circle(circ_center, r, fill=False, ec=color, lw=lw, zorder=3))
    ax.plot(*stem.T, color=color, lw=lw, zorder=3)

    in_pt = (stubs[0, 0], stubs[0, 1])
    out_pt = (stubs[3, 0], stubs[3, 1])
    return (cx - s * 0.7, cy - r, cx + s * 0.7, cy + r * 1.5), in_pt, out_pt


def draw_check_valve(
    ax, cx: float, cy: float, size: float = 0.45,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Triangle pointing in flow direction with a vertical bar."""
    h = size / 2
    s = size

    tri = np.array([[cx - h, cy + h], [cx + h, cy], [cx - h, cy - h]])
    bar = np.array([[cx + h, cy + h * 0.85], [cx + h, cy - h * 0.85]])
    lstub = np.array([[cx - s * 0.7, cy], [cx - h, cy]])
    rstub = np.array([[cx + h, cy], [cx + s * 0.7, cy]])

    if rotation:
        tri = _rot(tri, rotation, cx, cy)
        bar = _rot(bar, rotation, cx, cy)
        lstub = _rot(lstub, rotation, cx, cy)
        rstub = _rot(rstub, rotation, cx, cy)

    ax.plot(*lstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot(*rstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, lw=lw, zorder=3))
    ax.plot(*bar.T, color=color, lw=lw * 1.6, zorder=4, solid_capstyle="butt")

    in_pt = (lstub[0, 0], lstub[0, 1])
    out_pt = (rstub[1, 0], rstub[1, 1])
    return (cx - s * 0.7, cy - h, cx + s * 0.7, cy + h), in_pt, out_pt


def draw_control_valve(
    ax, cx: float, cy: float, size: float = 0.45,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8,
    fail: str = "FC"
) -> Tuple[BBox, Point, Point]:
    """Gate-valve body + actuator circle on stem (fail-state labelled)."""
    h = size / 2
    s = size
    act_r = size * 0.32
    act_cy = cy + h + act_r * 1.25

    ltri = np.array([[cx - h, cy + h], [cx - h, cy - h], [cx, cy]])
    rtri = np.array([[cx + h, cy + h], [cx + h, cy - h], [cx, cy]])
    stem = np.array([[cx, cy + h], [cx, act_cy - act_r]])
    lstub = np.array([[cx - s * 0.8, cy], [cx - h, cy]])
    rstub = np.array([[cx + h, cy], [cx + s * 0.8, cy]])
    act_c = np.array([cx, act_cy])

    if rotation:
        ltri = _rot(ltri, rotation, cx, cy)
        rtri = _rot(rtri, rotation, cx, cy)
        stem = _rot(stem, rotation, cx, cy)
        lstub = _rot(lstub, rotation, cx, cy)
        rstub = _rot(rstub, rotation, cx, cy)
        act_c = _rot(act_c.reshape(1, 2), rotation, cx, cy)[0]

    ax.plot(*lstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot(*rstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(Polygon(ltri, closed=True, fc=color, ec=color, lw=lw, zorder=3))
    ax.add_patch(Polygon(rtri, closed=True, fc=color, ec=color, lw=lw, zorder=3))
    ax.plot(*stem.T, color=color, lw=lw, zorder=3)
    ax.add_patch(Circle(act_c, act_r, fc="white", ec=color, lw=lw, zorder=4))
    ax.text(act_c[0], act_c[1], fail, ha="center", va="center",
            fontsize=5.5, color=color, zorder=5, fontfamily="monospace", fontweight="bold")

    in_pt = (lstub[0, 0], lstub[0, 1])
    out_pt = (rstub[1, 0], rstub[1, 1])
    top_y = act_cy + act_r if rotation == 0 else cy + s * 2
    return (cx - s * 0.8, cy - h, cx + s * 0.8, top_y), in_pt, out_pt


def draw_ball_valve(
    ax, cx: float, cy: float, size: float = 0.45,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Circle with T-handle."""
    r = size / 2
    s = size

    lstub = np.array([[cx - s * 0.7, cy], [cx - r, cy]])
    rstub = np.array([[cx + r, cy], [cx + s * 0.7, cy]])
    stem = np.array([[cx, cy + r], [cx, cy + r + s * 0.2]])
    handle = np.array([[cx - r * 0.5, cy + r + s * 0.2], [cx + r * 0.5, cy + r + s * 0.2]])

    if rotation:
        lstub = _rot(lstub, rotation, cx, cy)
        rstub = _rot(rstub, rotation, cx, cy)
        stem = _rot(stem, rotation, cx, cy)
        handle = _rot(handle, rotation, cx, cy)

    ax.plot(*lstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot(*rstub.T, color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=color, lw=lw, zorder=3))
    ax.plot(*stem.T, color=color, lw=lw, zorder=4)
    ax.plot(*handle.T, color=color, lw=lw * 1.5, zorder=4, solid_capstyle="butt")

    in_pt = (lstub[0, 0], lstub[0, 1])
    out_pt = (rstub[1, 0], rstub[1, 1])
    return (cx - s * 0.7, cy - r, cx + s * 0.7, cy + r + s * 0.2 + r * 0.5), in_pt, out_pt


def draw_centrifugal_pump(
    ax, cx: float, cy: float, r: float = 0.5,
    rotation: float = 0, color: Color = "#1a1a1a", lw: float = 1.8,
    tag: str = ""
) -> Tuple[BBox, Point, Point]:
    """Circle casing with inlet triangle and interior impeller arrow."""
    tri_size = r * 0.38
    stub_len = r * 0.8

    # Suction triangle (pointing into casing, left side)
    tri = np.array([
        [cx - r, cy],
        [cx - r - tri_size, cy + tri_size],
        [cx - r - tri_size, cy - tri_size],
    ])
    # Discharge stub (right)
    dis_in = np.array([cx + r, cy])
    dis_out = np.array([cx + r + stub_len, cy])
    # Suction stub connects up to triangle left vertex
    suc_in = np.array([cx - r - tri_size - stub_len, cy])
    suc_out = np.array([cx - r - tri_size, cy])

    if rotation:
        tri = _rot(tri, rotation, cx, cy)
        dis_in = _rot(dis_in.reshape(1, 2), rotation, cx, cy)[0]
        dis_out = _rot(dis_out.reshape(1, 2), rotation, cx, cy)[0]
        suc_in = _rot(suc_in.reshape(1, 2), rotation, cx, cy)[0]
        suc_out = _rot(suc_out.reshape(1, 2), rotation, cx, cy)[0]

    ax.add_patch(Circle((cx, cy), r, fc="#f8f8f8", ec=color, lw=lw, zorder=3))
    ax.add_patch(Polygon(tri, closed=True, fc=color, ec=color, zorder=4))
    ax.plot([suc_in[0], suc_out[0]], [suc_in[1], suc_out[1]], color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot([dis_in[0], dis_out[0]], [dis_in[1], dis_out[1]], color=color, lw=lw, zorder=2, solid_capstyle="butt")

    # Impeller arc
    theta = np.linspace(np.radians(20), np.radians(200), 30)
    ax.plot(cx + r * 0.55 * np.cos(theta), cy + r * 0.55 * np.sin(theta),
            color=color, lw=1.0, zorder=5)

    if tag:
        ax.text(cx, cy - r - 0.18, tag, ha="center", va="top", fontsize=6,
                color=color, zorder=5, fontfamily="monospace")

    in_pt = (suc_in[0], suc_in[1])
    out_pt = (dis_out[0], dis_out[1])
    x_min = min(suc_in[0], cx - r)
    x_max = max(dis_out[0], cx + r)
    return (x_min, cy - r, x_max, cy + r), in_pt, out_pt


def draw_tank(
    ax, cx: float, cy: float, w: float = 1.8, h: float = 4.5,
    tag: str = "", color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Flat-bottom cylindrical tank with domed top."""
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    dome_h = w * 0.18

    # Body sides
    ax.plot([x1, x1], [y1, y2 - dome_h], color=color, lw=lw, zorder=3)
    ax.plot([x2, x2], [y1, y2 - dome_h], color=color, lw=lw, zorder=3)
    # Flat bottom
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, zorder=3)
    # Domed top arc
    t = np.linspace(0, np.pi, 60)
    ax.plot(cx + (w / 2) * np.cos(t), (y2 - dome_h) + dome_h * np.sin(t),
            color=color, lw=lw, zorder=3)
    # Elliptic seam line
    ax.plot(cx + (w / 2) * np.cos(np.pi - t), (y2 - dome_h) - dome_h * 0.2 * np.sin(t),
            color=color, lw=0.6, linestyle="--", zorder=3, alpha=0.5)
    # Liquid level dashed line
    liq_y = y1 + h * 0.62
    ax.plot([x1 + 0.06, x2 - 0.06], [liq_y, liq_y],
            color=color, lw=0.7, linestyle="--", zorder=3, alpha=0.45)

    if tag:
        ax.text(cx, cy + h * 0.08, tag, ha="center", va="center", fontsize=7,
                color=color, zorder=5, fontfamily="monospace", fontweight="bold")

    in_pt = (x1, cy)
    out_pt = (x2, cy)
    return (x1, y1, x2, y2), in_pt, out_pt


def draw_vessel(
    ax, cx: float, cy: float, w: float = 1.2, h: float = 4.5,
    tag: str = "", color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Pressure vessel with elliptical heads."""
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    head_h = w * 0.28

    ax.plot([x1, x1], [y1 + head_h, y2 - head_h], color=color, lw=lw, zorder=3)
    ax.plot([x2, x2], [y1 + head_h, y2 - head_h], color=color, lw=lw, zorder=3)

    t = np.linspace(0, np.pi, 60)
    # Top head
    ax.plot(cx + (w / 2) * np.cos(t), (y2 - head_h) + head_h * np.sin(t),
            color=color, lw=lw, zorder=3)
    # Bottom head
    ax.plot(cx + (w / 2) * np.cos(t), (y1 + head_h) - head_h * np.sin(t),
            color=color, lw=lw, zorder=3)

    # Top nozzle
    ax.plot([cx, cx], [y2 - head_h + head_h, y2 + 0.28],
            color=color, lw=lw * 0.85, zorder=3)
    # Bottom nozzle
    ax.plot([cx, cx], [y1 - 0.28, y1 + head_h - head_h],
            color=color, lw=lw * 0.85, zorder=3)

    if tag:
        ax.text(cx, cy, tag, ha="center", va="center", fontsize=7,
                color=color, zorder=5, fontfamily="monospace", fontweight="bold", rotation=90)

    in_pt = (x1, cy)
    out_pt = (x2, cy)
    return (x1, y1 - 0.28, x2, y2 + 0.28), in_pt, out_pt


def draw_heat_exchanger(
    ax, cx: float, cy: float, w: float = 2.8, h: float = 1.1,
    tag: str = "", color: Color = "#1a1a1a", lw: float = 1.8
) -> Tuple[BBox, Point, Point]:
    """Shell-and-tube heat exchanger with shell and tube-side nozzles."""
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    cap_r = h / 2

    # Shell body
    ax.plot([x1 + cap_r * 0.4, x2 - cap_r * 0.4], [y1, y1], color=color, lw=lw, zorder=3)
    ax.plot([x1 + cap_r * 0.4, x2 - cap_r * 0.4], [y2, y2], color=color, lw=lw, zorder=3)

    # Left cap (left-pointing semicircle)
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 40)
    ax.plot(x1 + cap_r * 0.4 + cap_r * np.cos(t), cy + cap_r * np.sin(t),
            color=color, lw=lw, zorder=3)

    # Right cap (right-pointing semicircle)
    t2 = np.linspace(-np.pi / 2, np.pi / 2, 40)
    ax.plot(x2 - cap_r * 0.4 + cap_r * np.cos(t2), cy + cap_r * np.sin(t2),
            color=color, lw=lw, zorder=3)

    # Tube bundle lines
    for i, ty in enumerate(np.linspace(y1 + h * 0.22, y2 - h * 0.22, 3)):
        ax.plot([x1 + cap_r * 0.4, x2 - cap_r * 0.4], [ty, ty],
                color=color, lw=0.75, alpha=0.55, zorder=3)

    # Shell nozzles (top-left inlet, bottom-right outlet)
    shell_in_x = cx - w * 0.22
    shell_out_x = cx + w * 0.22
    ax.plot([shell_in_x, shell_in_x], [y2, y2 + 0.3], color=color, lw=lw * 0.8, zorder=3)
    ax.plot([shell_out_x, shell_out_x], [y1 - 0.3, y1], color=color, lw=lw * 0.8, zorder=3)

    # Tube-side stubs (process fluid, main line connection)
    ax.plot([x1 - 0.25, x1 + cap_r * 0.4 - cap_r], [cy, cy],
            color=color, lw=lw, zorder=2, solid_capstyle="butt")
    ax.plot([x2 - cap_r * 0.4 + cap_r, x2 + 0.25], [cy, cy],
            color=color, lw=lw, zorder=2, solid_capstyle="butt")

    if tag:
        ax.text(cx, cy, tag, ha="center", va="center", fontsize=6.5,
                color=color, zorder=5, fontfamily="monospace", fontweight="bold")

    in_pt = (x1 - 0.25, cy)
    out_pt = (x2 + 0.25, cy)
    return (x1 - 0.25, y1 - 0.3, x2 + 0.25, y2 + 0.3), in_pt, out_pt


def draw_instrument(
    ax, cx: float, cy: float, size: float = 0.52,
    tag: str = "PI", line_type: str = "solid",
    color: Color = "#1a1a1a", lw: float = 1.4,
    connect_to: Point | None = None
) -> BBox:
    """ISA instrument bubble (circle with tag)."""
    r = size / 2

    ls_map = {"solid": "-", "dashed": "--", "dotdash": "-."}
    ls = ls_map.get(line_type, "-")

    fc = "white" if line_type != "dotdash" else "#e8f4fd"
    circle = Circle((cx, cy), r, fc=fc, ec=color, lw=lw, linestyle=ls, zorder=5)
    ax.add_patch(circle)

    # Horizontal dividing line for multi-letter tags
    if len(tag) > 2:
        ax.plot([cx - r * 0.72, cx + r * 0.72], [cy, cy],
                color=color, lw=0.6, zorder=6)
        ax.text(cx, cy + r * 0.3, tag[:2], ha="center", va="center",
                fontsize=5.2, color=color, zorder=7, fontfamily="monospace", fontweight="bold")
        ax.text(cx, cy - r * 0.3, tag[2:], ha="center", va="center",
                fontsize=5.2, color=color, zorder=7, fontfamily="monospace", fontweight="bold")
    else:
        ax.text(cx, cy, tag, ha="center", va="center",
                fontsize=6.2, color=color, zorder=7, fontfamily="monospace", fontweight="bold")

    if connect_to:
        x2, y2 = connect_to
        ax.plot([cx, x2], [cy, y2], color=color, lw=0.9, linestyle=ls, zorder=4)
        # Small tee on pipe
        perp_len = 0.12
        dx, dy = x2 - cx, y2 - cy
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            nx, ny = -dy / length, dx / length
            ax.plot([x2 - nx * perp_len, x2 + nx * perp_len],
                    [y2 - ny * perp_len, y2 + ny * perp_len],
                    color=color, lw=lw * 0.8, zorder=4)

    return (cx - r, cy - r, cx + r, cy + r)


def draw_title_block(ax, title: str, drawing_no: str, revision: str = "A",
                     date: str = "", color: Color = "#1a1a1a"):
    """Draw title block in bottom-right corner."""
    fig = ax.get_figure()
    fw = ax.get_xlim()[1]
    fh = ax.get_ylim()[1]

    bw, bh = 7.0, 1.6
    bx, by = fw - bw - 0.2, 0.15

    ax.add_patch(mpatches.FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="square,pad=0",
        fc="white", ec=color, lw=1.2, zorder=8
    ))

    # Internal grid lines
    ax.plot([bx + bw * 0.5, bx + bw * 0.5], [by, by + bh], color=color, lw=0.7, zorder=9)
    ax.plot([bx + bw * 0.5, bx + bw], [by + bh * 0.55, by + bh * 0.55], color=color, lw=0.7, zorder=9)

    ax.text(bx + bw * 0.25, by + bh * 0.6, title, ha="center", va="center",
            fontsize=8.5, color=color, zorder=9, fontweight="bold")
    ax.text(bx + bw * 0.25, by + bh * 0.22, "PIPING & INSTRUMENTATION DIAGRAM",
            ha="center", va="center", fontsize=5.5, color=color, zorder=9)
    ax.text(bx + bw * 0.75, by + bh * 0.77, f"DWG NO: {drawing_no}",
            ha="center", va="center", fontsize=6.5, color=color, zorder=9, fontweight="bold")
    ax.text(bx + bw * 0.75, by + bh * 0.35, f"REV: {revision}   DATE: {date}",
            ha="center", va="center", fontsize=5.5, color=color, zorder=9)
