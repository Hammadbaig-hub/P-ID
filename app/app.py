"""
P&ID Intelligence Platform — Streamlit Application
Run: streamlit run app/app.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P&ID Intelligence Platform",
    page_icon="assets/icon.png" if Path("assets/icon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Enterprise CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background: #f4f6f9;
}

/* ── Top bar ── */
.top-bar {
    background: #0a1628;
    padding: 0.7rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-bottom: 3px solid #0696D7;
}
.top-bar-icon {
    width: 32px;
    height: 32px;
    background: #0696D7;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.top-bar-icon svg {
    width: 18px;
    height: 18px;
    fill: #ffffff;
}
.top-bar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.3px;
    line-height: 1.2;
}
.top-bar-subtitle {
    font-size: 0.72rem;
    color: #7a96bb;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.top-bar-badge {
    margin-left: auto;
    background: rgba(6,150,215,0.15);
    border: 1px solid #0696D7;
    color: #0696D7;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.5px;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-top: 3px solid #0696D7;
    border-radius: 4px;
    padding: 1.1rem 1rem 0.9rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(10,22,40,0.06);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0a1628;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #5d7292;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Status pills ── */
.status-ok {
    background: #e6f4ea;
    color: #1a7340;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 0.73rem;
    font-weight: 600;
    border: 1px solid #a8d5b5;
    letter-spacing: 0.3px;
}
.status-err {
    background: #fdf0f0;
    color: #8b2020;
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 0.73rem;
    font-weight: 600;
    border: 1px solid #f0c0c0;
    letter-spacing: 0.3px;
}

/* ── Sidebar override ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dde3ec;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #0a1628;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #5d7292;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #dde3ec;
}

/* ── Chat bubbles ── */
.chat-user {
    background: #eef4fb;
    border-left: 3px solid #0696D7;
    border-radius: 2px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}
.chat-assistant {
    background: #f8f9fb;
    border-left: 3px solid #0a1628;
    border-radius: 2px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}

/* ── Suggestion buttons ── */
.stButton > button {
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 3px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #dde3ec;
    background: transparent;
}
[data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    padding: 0.55rem 1.2rem;
    color: #5d7292;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #0696D7;
    border-bottom: 2px solid #0696D7;
}

/* ── Professional footer ── */
.pro-footer {
    margin-top: 3rem;
    padding: 1rem 1.5rem;
    background: #0a1628;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.pro-footer-left {
    font-size: 0.72rem;
    color: #7a96bb;
    font-weight: 500;
}
.pro-footer-right {
    font-size: 0.68rem;
    color: #4a6280;
    letter-spacing: 0.3px;
}
.pro-footer span {
    color: #0696D7;
    font-weight: 600;
}

/* ── Info box ── */
.pro-info {
    background: #f0f7ff;
    border-left: 3px solid #0696D7;
    border-radius: 2px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #0a1628;
}

/* ── Subheader override ── */
h3, .stSubheader {
    color: #0a1628;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
}

/* ── Dataframe / table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid #dde3ec;
    border-radius: 3px;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: transparent;
    border: 1px solid #0696D7;
    color: #0696D7;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 3px;
    padding: 0.3rem 0.8rem;
}
.stDownloadButton > button:hover {
    background: #0696D7;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ── API key ───────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv("ANTHROPIC_API_KEY", "")


# ── Lazy pipeline import ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import Pipeline
    return Pipeline(api_key=_get_api_key() or None)


# ── Session state defaults ────────────────────────────────────────────────
def _init_state():
    defaults = {"result": None, "messages": [], "pipeline_ready": False}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Professional top bar ──────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10
                     10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/>
        </svg>
    </div>
    <div>
        <div class="top-bar-title">P&amp;ID Intelligence Platform</div>
        <div class="top-bar-subtitle">Symbol Detection · OCR · Knowledge Graph · AI Assistant</div>
    </div>
    <div class="top-bar-badge">ISA S5.1 · ISO 10628</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Control Panel")

    st.markdown('<div class="section-label">Input Source</div>', unsafe_allow_html=True)
    mode = st.radio("Source", ["Generate Sample P&ID", "Upload Image"], index=0,
                    label_visibility="collapsed")

    uploaded = None
    seed_val = 42
    if mode == "Generate Sample P&ID":
        seed_val = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    else:
        uploaded = st.file_uploader(
            "Upload P&ID image", type=["png", "jpg", "jpeg", "tif", "bmp"],
            help="PNG / JPEG / TIFF accepted"
        )

    st.divider()

    st.markdown('<div class="section-label">Detection Settings</div>', unsafe_allow_html=True)
    with st.expander("Configure Thresholds"):
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

    st.markdown("")
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()

    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)

    if st.session_state.result:
        r = st.session_state.result
        st.markdown('<span class="status-ok">Analysis Complete</span>', unsafe_allow_html=True)
        st.caption(f"{r.num_symbols} symbols  ·  {r.num_text_regions} text regions")
        if r.errors:
            with st.expander(f"{len(r.errors)} warning(s)"):
                for e in r.errors:
                    st.caption(e)
    else:
        st.markdown('<span class="status-err">No Analysis Run</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<small style='color:#5d7292;line-height:1.6'>"
        "P&amp;ID Intelligence Platform v1.0<br>"
        "ISA S5.1 · ISO 10628 compliant</small>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    pipe = get_pipeline()
    pipe.cfg.setdefault("detection", {})["confidence"] = conf_thresh
    pipe.detector.conf = conf_thresh

    with st.spinner("Running P&ID analysis pipeline..."):
        prog = st.progress(0, text="Initialising...")
        try:
            if mode == "Generate Sample P&ID":
                prog.progress(15, "Generating synthetic P&ID...")
                result = pipe.run_generated(seed=int(seed_val), save_dir="images/processed")
            else:
                if uploaded is None:
                    st.error("Please upload an image first.")
                    st.stop()
                prog.progress(15, "Loading image...")
                img = Image.open(uploaded).convert("RGB")
                result = pipe.run_from_image(img)

            prog.progress(50, "Detection complete...")
            prog.progress(75, "Building knowledge graph...")
            prog.progress(95, "Loading AI context...")
            prog.progress(100, "Done")

            st.session_state.result = result
            st.session_state.pipeline_ready = True
            st.session_state.messages = []

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())
        finally:
            prog.empty()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_dash, tab_detect, tab_graph, tab_chat = st.tabs(
    ["Dashboard", "Detection & OCR", "Knowledge Graph", "AI Assistant"]
)

result = st.session_state.result

# Professional chart color palette
_NAVY_SCALE = ["#0a1628", "#0d2241", "#0f3460", "#0696D7", "#45b4e8", "#8dd3f7"]
_PIE_COLORS = ["#0a1628", "#0696D7", "#4a90c4"]


# ─── TAB 1: DASHBOARD ────────────────────────────────────────────────────
with tab_dash:
    if result is None:
        st.markdown("""
        <div class="pro-info">
            Select an input source in the Control Panel and click <strong>Run Analysis</strong> to begin.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        _col1, _col2 = st.columns(2)
        with _col1:
            st.markdown("""
            #### Platform Capabilities
            1. **Generate** realistic P&ID diagrams with ISA-compliant symbols
            2. **Detect** equipment, valves, and instruments using YOLOv8
            3. **Extract** tag numbers and labels with PaddleOCR
            4. **Build** a knowledge graph with NetworkX
            5. **Chat** with an AI engineer about your P&ID
            """)
        with _col2:
            st.markdown("""
            #### Detected Symbol Classes
            | Category | Symbols |
            |----------|---------|
            | Equipment | Tank, Vessel, Column, Pump, Heat Exchanger |
            | Valves | Gate, Globe, Check, Control, Ball, Butterfly |
            | Instruments | Pressure, Temperature, Flow, Level, Analyser |
            """)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{result.num_symbols}</div>
                <div class="metric-label">Symbols Detected</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{result.num_text_regions}</div>
                <div class="metric-label">Text Regions (OCR)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{result.num_graph_nodes}</div>
                <div class="metric-label">Graph Nodes</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{result.num_graph_edges}</div>
                <div class="metric-label">Graph Edges</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        img_col, chart_col = st.columns([2, 1])
        with img_col:
            st.subheader("Detected Symbols")
            st.image(result.annotated_image, use_container_width=True,
                     caption="Bounding boxes coloured by class")

        with chart_col:
            st.subheader("Symbol Distribution")
            if result.detections:
                cls_counts = {}
                for d in result.detections:
                    cls_counts[d.class_name] = cls_counts.get(d.class_name, 0) + 1
                df_cls = pd.DataFrame(
                    [{"Class": k, "Count": v} for k, v in sorted(
                        cls_counts.items(), key=lambda x: -x[1]
                    )]
                )
                fig_bar = px.bar(
                    df_cls, x="Count", y="Class", orientation="h",
                    color="Count",
                    color_continuous_scale=[[0, "#0a1628"], [0.5, "#0696D7"], [1, "#8dd3f7"]],
                    height=350,
                )
                fig_bar.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#0a1628", size=11),
                    coloraxis_showscale=False,
                )
                fig_bar.update_xaxes(gridcolor="#eaecf0", zeroline=False)
                fig_bar.update_yaxes(gridcolor="#eaecf0")
                st.plotly_chart(fig_bar, use_container_width=True)

                cat_map = {
                    "Equipment": {"tank", "vessel", "column", "heat_exchanger", "centrifugal_pump"},
                    "Valves":    {"gate_valve", "globe_valve", "check_valve", "control_valve",
                                  "ball_valve", "butterfly_valve"},
                    "Instruments": {"pressure_instrument", "temperature_instrument",
                                    "flow_instrument", "level_instrument", "analyzer_instrument"},
                }
                cat_counts = {"Equipment": 0, "Valves": 0, "Instruments": 0}
                for cls, cnt in cls_counts.items():
                    for cat, names in cat_map.items():
                        if cls in names:
                            cat_counts[cat] += cnt
                df_cat = pd.DataFrame({"Category": list(cat_counts.keys()),
                                        "Count": list(cat_counts.values())})
                fig_pie = px.pie(
                    df_cat, values="Count", names="Category",
                    color_discrete_sequence=_PIE_COLORS,
                    height=200,
                )
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", font=dict(size=10)),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#0a1628"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)


# ─── TAB 2: DETECTION & OCR ──────────────────────────────────────────────
with tab_detect:
    if result is None:
        st.markdown('<div class="pro-info">Run the analysis first.</div>',
                    unsafe_allow_html=True)
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Symbol Detections")
            if result.detections:
                det_df = pd.DataFrame(result.detection_dicts)[
                    ["tag", "class_name", "description", "confidence"]
                ]
                det_df.columns = ["Tag", "Class", "Description", "Confidence"]
                det_df["Confidence"] = det_df["Confidence"].map("{:.2%}".format)
                st.dataframe(det_df, use_container_width=True, height=380)
                csv = det_df.to_csv(index=False)
                st.download_button("Download Detections CSV", csv,
                                   "detections.csv", "text/csv")
            else:
                st.warning("No symbols detected.")

        with right:
            st.subheader("OCR Extracted Text")
            if result.ocr_regions:
                ocr_df = pd.DataFrame(result.ocr_dicts)[
                    ["text", "confidence", "is_tag", "tag_type"]
                ]
                ocr_df.columns = ["Text", "Confidence", "Is Tag", "Tag Type"]
                ocr_df["Confidence"] = ocr_df["Confidence"].map("{:.2%}".format)
                def _highlight(row):
                    return ["background-color: #eef4fb" if row["Is Tag"] else ""
                            for _ in row]
                st.dataframe(
                    ocr_df.style.apply(_highlight, axis=1),
                    use_container_width=True, height=380
                )
                csv2 = ocr_df.to_csv(index=False)
                st.download_button("Download OCR CSV", csv2, "ocr.csv", "text/csv")
            else:
                st.warning("No text extracted.")

        st.divider()
        st.subheader("Image Comparison")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.caption("Original")
            st.image(result.image, use_container_width=True)
        with oc2:
            st.caption("Annotated (detections + OCR)")
            annotated = result.annotated_image
            if result.ocr_regions:
                annotated = get_pipeline().ocr.annotate(annotated, result.ocr_regions)
            st.image(annotated, use_container_width=True)


# ─── TAB 3: KNOWLEDGE GRAPH ──────────────────────────────────────────────
with tab_graph:
    if result is None or result.graph is None:
        st.markdown('<div class="pro-info">Run the analysis first.</div>',
                    unsafe_allow_html=True)
    else:
        from src.graph.visualizer import GraphVisualizer
        viz = GraphVisualizer()
        summary = viz.summary(result.graph)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Nodes", summary.get("nodes", 0))
        g2.metric("Edges", summary.get("edges", 0))
        g3.metric("Connected Components", summary.get("connected_components", 0))
        g4.metric("Is DAG", "Yes" if summary.get("is_dag") else "No")

        fig_graph = viz.plot(result.graph, title="P&ID Equipment Knowledge Graph")
        st.plotly_chart(fig_graph, use_container_width=True)

        with st.expander("Node Details Table"):
            node_rows = []
            for n, data in result.graph.nodes(data=True):
                node_rows.append({
                    "Tag": n,
                    "Class": data.get("class_name", ""),
                    "Description": data.get("description", ""),
                    "Connections Out": result.graph.out_degree(n),
                    "Connections In":  result.graph.in_degree(n),
                })
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True)

        with st.expander("Edge (Connection) Details"):
            edge_rows = []
            for u, v, data in result.graph.edges(data=True):
                edge_rows.append({
                    "From": u, "To": v,
                    "Stream": data.get("stream", ""),
                    "Fluid": data.get("fluid", ""),
                })
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)

        with st.expander("Export Graph"):
            import json as _json
            st.download_button(
                "Download JSON", _json.dumps(result.graph_json, indent=2),
                "pid_graph.json", "application/json"
            )
            try:
                import io
                buf = io.StringIO()
                import networkx as nx
                H = result.graph.copy()
                for n in H.nodes():
                    if "bbox" in H.nodes[n]:
                        H.nodes[n]["bbox"] = str(H.nodes[n]["bbox"])
                nx.write_graphml(H, buf)
                st.download_button("Download GraphML", buf.getvalue(),
                                   "pid_graph.graphml", "application/xml")
            except Exception:
                pass


# ─── TAB 4: AI ASSISTANT ─────────────────────────────────────────────────
with tab_chat:
    if result is None:
        st.markdown('<div class="pro-info">Run the analysis first, then chat about your P&ID.</div>',
                    unsafe_allow_html=True)
    else:
        pipe = get_pipeline()

        st.markdown('<div class="section-label">Suggested Questions</div>',
                    unsafe_allow_html=True)
        suggestions = pipe.assistant.suggested_questions()
        cols = st.columns(2)
        for i, q in enumerate(suggestions[:6]):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Processing..."):
                    response = pipe.chat(q)
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.divider()

        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about this P&ID diagram..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    text_holder = st.empty()
                    full = ""
                    try:
                        for chunk in pipe.chat_stream(prompt):
                            full += chunk
                            text_holder.markdown(full + "▌")
                        text_holder.markdown(full)
                    except Exception as e:
                        full = f"Error: {e}"
                        text_holder.markdown(full)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full}
                    )

        if st.session_state.messages:
            if st.button("Clear Conversation", key="clear_chat"):
                st.session_state.messages = []
                pipe.assistant.clear_history()
                st.rerun()


# ── Professional footer ───────────────────────────────────────────────────
st.markdown("""
<div class="pro-footer">
    <div class="pro-footer-left">
        <span>P&amp;ID Intelligence Platform</span> &nbsp;v1.0.0
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Powered by YOLOv8 · PaddleOCR · NetworkX · Claude AI
    </div>
    <div class="pro-footer-right">
        ISA S5.1 Compliant &nbsp;·&nbsp; ISO 10628 Compliant &nbsp;·&nbsp; 2024
    </div>
</div>
""", unsafe_allow_html=True)
