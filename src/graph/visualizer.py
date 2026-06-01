"""
Graph Visualizer
Renders a NetworkX DiGraph as an interactive Plotly figure.
"""

import math
import networkx as nx
import plotly.graph_objects as go
from plotly.graph_objects import Figure


INSTRUMENT_CLASSES = {
    "pressure_instrument", "temperature_instrument",
    "flow_instrument", "level_instrument", "analyzer_instrument",
}

VALVE_CLASSES = {
    "gate_valve", "globe_valve", "check_valve", "control_valve",
    "ball_valve", "butterfly_valve",
}

EQUIPMENT_CLASSES = {"tank", "vessel", "column", "heat_exchanger", "centrifugal_pump"}


def _layout(G: nx.DiGraph) -> dict:
    """Layered layout: equipment left-to-right, instruments above/below."""
    process_nodes = [n for n, d in G.nodes(data=True)
                     if d.get("class_name") in EQUIPMENT_CLASSES]
    valve_nodes   = [n for n, d in G.nodes(data=True)
                     if d.get("class_name") in VALVE_CLASSES]
    instr_nodes   = [n for n, d in G.nodes(data=True)
                     if d.get("class_name") in INSTRUMENT_CLASSES]
    other         = [n for n in G.nodes()
                     if n not in process_nodes + valve_nodes + instr_nodes]

    # Sort process nodes by bbox x
    process_nodes.sort(
        key=lambda n: G.nodes[n].get("bbox", [0, 0, 0, 0])[0]
    )

    pos = {}
    # Main process row at y=0
    for i, n in enumerate(process_nodes):
        pos[n] = (i * 2.5, 0)

    # Valves interpolated between process nodes
    for i, n in enumerate(valve_nodes):
        pos[n] = (i * 2.5 / max(len(valve_nodes), 1) * len(process_nodes) * 2.5
                  + 1.2, 0.0)

    # Instruments above or below alternately
    for i, n in enumerate(instr_nodes):
        y = 1.8 if i % 2 == 0 else -1.8
        x_anchor = pos.get(process_nodes[i % max(len(process_nodes), 1)], (i, 0))[0]
        pos[n] = (x_anchor, y)

    # Fallback spring layout for anything missing
    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        sub = G.subgraph(missing)
        spring = nx.spring_layout(sub, seed=42)
        offset_x = max((p[0] for p in pos.values()), default=0) + 3
        for n, (x, y) in spring.items():
            pos[n] = (x + offset_x, y)

    return pos


def build_plotly_figure(G: nx.DiGraph,
                        title: str = "P&ID Knowledge Graph",
                        height: int = 520) -> Figure:
    """Return an interactive Plotly figure of the knowledge graph."""
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No graph data yet", height=height)
        return fig

    pos = _layout(G)

    # ── Edge traces ───────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    edge_label_x, edge_label_y, edge_labels = [], [], []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label_x.append(mx)
        edge_label_y.append(my)
        stream = data.get("stream", "")
        fluid  = data.get("fluid", "")
        edge_labels.append(f"{stream}<br>{fluid}" if fluid else stream)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.8, color="#888"),
        hoverinfo="none",
        showlegend=False,
    )

    edge_label_trace = go.Scatter(
        x=edge_label_x, y=edge_label_y,
        mode="text",
        text=edge_labels,
        textfont=dict(size=8, color="#555"),
        hoverinfo="none",
        showlegend=False,
    )

    # Arrow annotations
    arrows = []
    for u, v in G.edges():
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        arrows.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor="#666",
        ))

    # ── Node traces (one per class group for legend) ──────────────────────
    groups: dict[str, dict] = {}
    for n, data in G.nodes(data=True):
        cls = data.get("class_name", "unknown")
        grp = "Equipment" if cls in EQUIPMENT_CLASSES else \
              "Valve"     if cls in VALVE_CLASSES else \
              "Instrument" if cls in INSTRUMENT_CLASSES else "Other"
        if grp not in groups:
            groups[grp] = {"x": [], "y": [], "text": [], "hover": [],
                           "color": [], "size": []}
        x, y = pos.get(n, (0, 0))
        groups[grp]["x"].append(x)
        groups[grp]["y"].append(y)
        groups[grp]["text"].append(n)
        desc = data.get("description", cls)
        conf = data.get("confidence", 0)
        groups[grp]["hover"].append(
            f"<b>{n}</b><br>{desc}<br>Class: {cls}<br>Conf: {conf:.2f}"
        )
        groups[grp]["color"].append(data.get("color", "#888"))
        groups[grp]["size"].append(data.get("size", 16))

    grp_colors = {"Equipment": "#4682B4", "Valve": "#2E8B57",
                  "Instrument": "#DAA520", "Other": "#888"}

    node_traces = []
    for grp, vals in groups.items():
        node_traces.append(go.Scatter(
            x=vals["x"], y=vals["y"],
            mode="markers+text",
            marker=dict(
                color=vals["color"],
                size=vals["size"],
                line=dict(width=1.5, color="white"),
                symbol="circle",
            ),
            text=vals["text"],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=vals["hover"],
            name=grp,
            showlegend=True,
        ))

    layout = go.Layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        annotations=arrows,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return go.Figure(data=[edge_trace, edge_label_trace, *node_traces], layout=layout)


class GraphVisualizer:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def plot(self, G: nx.DiGraph, title: str = "P&ID Knowledge Graph") -> Figure:
        return build_plotly_figure(G, title=title)

    def summary(self, G: nx.DiGraph) -> dict:
        if G.number_of_nodes() == 0:
            return {}
        classes = {}
        for _, data in G.nodes(data=True):
            c = data.get("class_name", "unknown")
            classes[c] = classes.get(c, 0) + 1
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "class_counts": classes,
            "connected_components": nx.number_weakly_connected_components(G),
        }
