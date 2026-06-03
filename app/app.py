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
    background: linear-gradient(90deg, #0a1628 0%, #0d1d35 100%);
    padding: 0.7rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-bottom: 2px solid #0696D7;
}
.top-bar-icon {
    width: 32px;
    height: 32px;
    background: #0696D7;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
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
    font-size: 0.71rem;
    color: #7a96bb;
    font-weight: 400;
    letter-spacing: 0.2px;
    margin-top: 2px;
}
.top-bar-badge {
    margin-left: auto;
    background: rgba(6,150,215,0.15);
    border: 1px solid rgba(6,150,215,0.6);
    color: #0696D7;
    font-size: 0.67rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
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

/* ── Chat message bubbles (st.chat_message overrides) ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.35rem 0 !important;
    gap: 0.7rem !important;
    align-items: flex-start !important;
}
[data-testid="stChatMessageContent"] {
    padding: 0.75rem 1.05rem !important;
    max-width: 78% !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* User bubble — right aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: #0a1628 !important;
    border-radius: 18px 4px 18px 18px !important;
    box-shadow: 0 2px 6px rgba(10,22,40,0.22) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] li {
    color: #e8f0fb !important;
    margin: 0 !important;
}

/* Assistant bubble — left aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #dde3ec !important;
    border-radius: 4px 18px 18px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    color: #0a1628 !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-user"] {
    background: #0696D7 !important;
    border-radius: 50% !important;
    width: 34px !important; height: 34px !important; min-width: 34px !important;
    flex-shrink: 0 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #0a1628 !important;
    border-radius: 50% !important;
    width: 34px !important; height: 34px !important; min-width: 34px !important;
    flex-shrink: 0 !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1.5px solid #c8d4e3 !important;
    background: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.2rem !important;
    resize: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #0696D7 !important;
    box-shadow: 0 0 0 3px rgba(6,150,215,0.12) !important;
    outline: none !important;
    background: #ffffff !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: #0696D7 !important;
    border: none !important;
    border-radius: 50% !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: #0579b0 !important;
}

/* ── Welcome / empty state ── */
.chat-welcome {
    text-align: center;
    padding: 2.5rem 1rem 1.2rem;
}
.chat-welcome-orb {
    width: 60px; height: 60px;
    background: #0a1628;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1.1rem;
    box-shadow: 0 4px 16px rgba(10,22,40,0.25);
}
.chat-welcome-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #0a1628 !important;
    margin-bottom: 0.35rem !important;
}
.chat-welcome-sub {
    font-size: 0.84rem;
    color: #5d7292;
    max-width: 460px;
    margin: 0 auto 1.8rem;
    line-height: 1.6;
}

/* Suggested question chip buttons */
[data-testid="baseButton-secondary"] {
    background: #f0f7ff !important;
    border: 1px solid #b8d4f0 !important;
    border-radius: 20px !important;
    color: #0a1628 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 0.9rem !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.4rem !important;
    line-height: 1.35 !important;
    transition: all 0.15s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: #0696D7 !important;
    border-color: #0696D7 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(6,150,215,0.25) !important;
}

/* Clear conversation subtle override */
.clear-btn-row [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #dde3ec !important;
    border-radius: 3px !important;
    color: #5d7292 !important;
    font-size: 0.75rem !important;
    padding: 0.25rem 0.7rem !important;
    min-height: 1.8rem !important;
    transform: none !important;
}
.clear-btn-row [data-testid="baseButton-secondary"]:hover {
    background: #fdf0f0 !important;
    border-color: #e8b0b0 !important;
    color: #8b2020 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Chips label */
.chips-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #5d7292;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    text-align: center;
    margin-bottom: 0.6rem;
}

/* Typing animation */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.1rem 0;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #0696D7;
    border-radius: 50%;
    animation: tdot 1.3s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.18s; }
.typing-dot:nth-child(3) { animation-delay: 0.36s; }
@keyframes tdot {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.35; }
    30% { transform: translateY(-5px); opacity: 1; }
}

/* ── Suggestion buttons (general) ── */
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
    padding: 0.9rem 1.6rem;
    background: linear-gradient(90deg, #0a1628 0%, #0d1d35 100%);
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    border-top: 2px solid #0696D7;
}
.pro-footer-left {
    font-size: 0.72rem;
    color: #7a96bb;
    font-weight: 500;
    letter-spacing: 0.1px;
}
.pro-footer-right {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.pro-footer-badge {
    font-size: 0.65rem;
    color: #5a7899;
    font-weight: 600;
    letter-spacing: 0.3px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 2px 8px;
    border-radius: 10px;
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
            <path d="M12 15.5a3.5 3.5 0 01-3.5-3.5A3.5 3.5 0 0112 8.5a3.5 3.5 0 013.5 3.5 3.5 3.5 0 01-3.5 3.5m7.43-2.92c.04-.3.07-.6.07-.93s-.03-.63-.07-.93l2-1.63c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54C14.44 2.17 14.24 2 14 2h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.2-.08.47.12.61l2 1.63c-.04.3-.06.62-.06.93s.02.63.06.93l-2 1.63c-.2.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.49.37 1.03.7 1.62.94l.36 2.54c.04.24.23.41.47.41H14c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.57 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.2.07-.47-.12-.61l-2-1.63z"/>
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
            2. **Detect** equipment, valves, and instruments using YOLO26
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
        suggestions = pipe.assistant.suggested_questions()
        has_messages = bool(st.session_state.messages)

        # ── Empty state: welcome orb + suggestion chips ───────────────────
        if not has_messages:
            st.markdown("""
            <div class="chat-welcome">
                <div class="chat-welcome-orb">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="white">
                        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                    </svg>
                </div>
                <div class="chat-welcome-title">P&amp;ID AI Assistant</div>
                <div class="chat-welcome-sub">
                    Ask questions about detected equipment, process flows, instrument tags,
                    or engineering specifications in your diagram.
                </div>
            </div>
            <p class="chips-label">Suggested Questions</p>
            """, unsafe_allow_html=True)

            chip_cols = st.columns(3)
            for i, q in enumerate(suggestions[:6]):
                if chip_cols[i % 3].button(q, key=f"chip_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner(""):
                        response = pipe.chat(q)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

            chat_container = st.container()

        # ── Conversation view ─────────────────────────────────────────────
        else:
            chat_container = st.container(height=480)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Clear button — right-aligned, subtle
            _, rcol = st.columns([6, 1])
            with rcol:
                st.markdown('<div class="clear-btn-row">', unsafe_allow_html=True)
                if st.button("Clear", key="clear_chat"):
                    st.session_state.messages = []
                    pipe.assistant.clear_history()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Chat input (always rendered) ──────────────────────────────────
        if prompt := st.chat_input("Ask anything about this P&ID diagram..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    text_holder = st.empty()
                    # Typing indicator until first token
                    text_holder.markdown("""<div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>""", unsafe_allow_html=True)
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
            st.rerun()


# ── Professional footer ───────────────────────────────────────────────────
st.markdown("""
<div class="pro-footer">
    <div class="pro-footer-left">
        <span>P&amp;ID Intelligence Platform</span> &nbsp;v1.0.0
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Powered by YOLO26 · PaddleOCR · NetworkX · Claude AI
    </div>
    <div class="pro-footer-right">
        <span class="pro-footer-badge">ISA S5.1</span>
        <span class="pro-footer-badge">ISO 10628</span>
        <span class="pro-footer-badge">2024</span>
    </div>
</div>
""", unsafe_allow_html=True)
