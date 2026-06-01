"""
P&ID Intelligence Platform — Streamlit Application
Run: streamlit run app/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# ── Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="P&ID Intelligence Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Header */
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}
.main-header h1 { color: white; margin: 0; font-size: 1.8rem; letter-spacing: 0.5px; }
.main-header p  { color: #b3d9ff; margin: 0.3rem 0 0; font-size: 0.85rem; }

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #0d47a1; }
.metric-label { font-size: 0.78rem; color: #666; margin-top: 0.2rem; }

/* Status pill */
.status-ok  { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.status-err { background:#ffebee; color:#c62828; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* Chat bubbles */
.chat-user      { background:#e3f2fd; border-radius:12px 12px 2px 12px; padding:0.8rem; margin:0.4rem 0; }
.chat-assistant { background:#f3e5f5; border-radius:12px 12px 12px 2px; padding:0.8rem; margin:0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ── API key: Streamlit secrets first, env var as local-dev fallback ───────
def _get_api_key() -> str:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv("ANTHROPIC_API_KEY", "")


# ── Lazy pipeline import (avoids import errors at startup) ────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import Pipeline
    return Pipeline(api_key=_get_api_key() or None)


# ── Session state defaults ────────────────────────────────────────────────
def _init_state():
    defaults = {
        "result": None,
        "messages": [],
        "pipeline_ready": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚙️ P&ID Intelligence Platform</h1>
    <p>AI-powered Piping & Instrumentation Diagram digitisation · Symbol Detection · OCR · Knowledge Graph · Chat</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    # ── Input ─────────────────────────────────────────────────────────────
    st.markdown("### 📥 Input")
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

    # ── Detection settings ────────────────────────────────────────────────
    with st.expander("🔍 Detection Settings"):
        conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)

    # ── Run button ────────────────────────────────────────────────────────
    run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### ℹ️ Pipeline Status")

    if st.session_state.result:
        r = st.session_state.result
        st.markdown(f'<span class="status-ok">✓ Analysis complete</span>', unsafe_allow_html=True)
        st.caption(f"{r.num_symbols} symbols  ·  {r.num_text_regions} text regions")
        if r.errors:
            with st.expander(f"⚠️ {len(r.errors)} warning(s)"):
                for e in r.errors:
                    st.caption(e)
    else:
        st.markdown('<span class="status-err">○ No analysis run yet</span>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<small>P&ID Intelligence Platform v1.0<br>"
        "ISA S5.1 · ISO 10628 compliant</small>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    pipe = get_pipeline()
    # Update conf threshold
    pipe.cfg.setdefault("detection", {})["confidence"] = conf_thresh
    pipe.detector.conf = conf_thresh

    with st.spinner("Running P&ID analysis pipeline …"):
        prog = st.progress(0, text="Initialising …")
        try:
            if mode == "Generate Sample P&ID":
                prog.progress(15, "Generating synthetic P&ID …")
                result = pipe.run_generated(seed=int(seed_val),
                                            save_dir="images/processed")
            else:
                if uploaded is None:
                    st.error("Please upload an image first.")
                    st.stop()
                prog.progress(15, "Loading image …")
                img = Image.open(uploaded).convert("RGB")
                result = pipe.run_from_image(img)

            prog.progress(50, "Detection complete …")
            prog.progress(75, "Building knowledge graph …")
            prog.progress(95, "Loading AI context …")
            prog.progress(100, "Done ✓")

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
    ["📊 Dashboard", "🔍 Detection & OCR", "🌐 Knowledge Graph", "🤖 AI Assistant"]
)

result = st.session_state.result


# ─── TAB 1: DASHBOARD ────────────────────────────────────────────────────
with tab_dash:
    if result is None:
        st.info("👈 Select a source and click **Run Analysis** to get started.")
        _col1, _col2 = st.columns(2)
        with _col1:
            st.markdown("""
            #### What this platform does
            1. **Generate** realistic P&ID diagrams with ISA-compliant symbols
            2. **Detect** equipment, valves, and instruments using YOLOv8
            3. **Extract** tag numbers and labels with PaddleOCR
            4. **Build** a knowledge graph with NetworkX
            5. **Chat** with an AI engineer about your P&ID
            """)
        with _col2:
            st.markdown("""
            #### Detected symbol classes
            | Category | Symbols |
            |----------|---------|
            | Equipment | Tank, Vessel, Column, Pump, Heat Exchanger |
            | Valves | Gate, Globe, Check, Control, Ball, Butterfly |
            | Instruments | Pressure, Temperature, Flow, Level, Analyser |
            """)
    else:
        # ── Metrics ──────────────────────────────────────────────────────
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

        # ── Annotated image ───────────────────────────────────────────────
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
                fig_bar = px.bar(df_cls, x="Count", y="Class", orientation="h",
                                  color="Count", color_continuous_scale="Blues",
                                  height=350)
                fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)

                # Class type breakdown
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
                fig_pie = px.pie(df_cat, values="Count", names="Category",
                                  color_discrete_sequence=px.colors.qualitative.Set2,
                                  height=200)
                fig_pie.update_layout(margin=dict(l=0, r=0, t=20, b=0),
                                       showlegend=True, legend=dict(orientation="h"))
                st.plotly_chart(fig_pie, use_container_width=True)


# ─── TAB 2: DETECTION & OCR ──────────────────────────────────────────────
with tab_detect:
    if result is None:
        st.info("Run the analysis first.")
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("🔍 Symbol Detections")
            if result.detections:
                det_df = pd.DataFrame(result.detection_dicts)[
                    ["tag", "class_name", "description", "confidence"]
                ]
                det_df.columns = ["Tag", "Class", "Description", "Confidence"]
                det_df["Confidence"] = det_df["Confidence"].map("{:.2%}".format)
                st.dataframe(det_df, use_container_width=True, height=380)

                # Download button
                csv = det_df.to_csv(index=False)
                st.download_button("⬇ Download detections CSV", csv,
                                   "detections.csv", "text/csv")
            else:
                st.warning("No symbols detected.")

        with right:
            st.subheader("📝 OCR Extracted Text")
            if result.ocr_regions:
                ocr_df = pd.DataFrame(result.ocr_dicts)[
                    ["text", "confidence", "is_tag", "tag_type"]
                ]
                ocr_df.columns = ["Text", "Confidence", "Is Tag", "Tag Type"]
                ocr_df["Confidence"] = ocr_df["Confidence"].map("{:.2%}".format)
                # Highlight tags
                def _highlight(row):
                    return ["background-color: #e8f5e9" if row["Is Tag"] else ""
                            for _ in row]
                st.dataframe(
                    ocr_df.style.apply(_highlight, axis=1),
                    use_container_width=True, height=380
                )
                csv2 = ocr_df.to_csv(index=False)
                st.download_button("⬇ Download OCR CSV", csv2, "ocr.csv", "text/csv")
            else:
                st.warning("No text extracted.")

        # ── Original vs Annotated side-by-side ────────────────────────────
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
                annotated = get_pipeline().ocr.annotate(
                    annotated, result.ocr_regions
                )
            st.image(annotated, use_container_width=True)


# ─── TAB 3: KNOWLEDGE GRAPH ──────────────────────────────────────────────
with tab_graph:
    if result is None or result.graph is None:
        st.info("Run the analysis first.")
    else:
        from src.graph.visualizer import GraphVisualizer
        viz = GraphVisualizer()
        summary = viz.summary(result.graph)

        # Summary metrics
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Nodes", summary.get("nodes", 0))
        g2.metric("Edges", summary.get("edges", 0))
        g3.metric("Connected Components", summary.get("connected_components", 0))
        g4.metric("Is DAG", "✓ Yes" if summary.get("is_dag") else "✗ No")

        # Interactive graph
        fig_graph = viz.plot(result.graph, title="P&ID Equipment Knowledge Graph")
        st.plotly_chart(fig_graph, use_container_width=True)

        # Node details table
        with st.expander("📋 Node Details Table"):
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

        # Edge details
        with st.expander("🔗 Edge (Connection) Details"):
            edge_rows = []
            for u, v, data in result.graph.edges(data=True):
                edge_rows.append({
                    "From": u, "To": v,
                    "Stream": data.get("stream", ""),
                    "Fluid": data.get("fluid", ""),
                })
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)

        # Export
        with st.expander("⬇ Export Graph"):
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
        st.info("Run the analysis first, then chat about your P&ID.")
    else:
        pipe = get_pipeline()

        # ── Suggested questions ───────────────────────────────────────────
        st.markdown("**💡 Suggested questions:**")
        suggestions = pipe.assistant.suggested_questions()
        cols = st.columns(2)
        for i, q in enumerate(suggestions[:6]):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking …"):
                    response = pipe.chat(q)
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.divider()

        # ── Chat history ──────────────────────────────────────────────────
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ── Input ─────────────────────────────────────────────────────────
        if prompt := st.chat_input("Ask anything about this P&ID diagram …"):
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
                        full = f"❌ {e}"
                        text_holder.markdown(full)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full}
                    )

        # ── Clear chat ────────────────────────────────────────────────────
        if st.session_state.messages:
            if st.button("🗑 Clear conversation", key="clear_chat"):
                st.session_state.messages = []
                pipe.assistant.clear_history()
                st.rerun()
