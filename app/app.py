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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: #f0f2f7; }
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
}

/* ── Top bar ── */
.top-bar {
    background: linear-gradient(135deg, #0a1628 0%, #0d2241 100%);
    padding: 0 2.5rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    height: 66px;
    border-bottom: 3px solid #0696D7;
    box-shadow: 0 4px 24px rgba(10,22,40,0.32), 0 1px 6px rgba(10,22,40,0.2);
    position: relative;
    z-index: 100;
}
.top-bar::after {
    content: '';
    position: absolute;
    bottom: -3px; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(6,150,215,0.6), transparent);
}
.top-bar-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #0696D7, #0579b0);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 3px 10px rgba(6,150,215,0.45);
    flex-shrink: 0;
}
.top-bar-icon svg { width: 20px; height: 20px; fill: #ffffff; }
.top-bar-title {
    font-size: 1.07rem; font-weight: 700; color: #ffffff;
    letter-spacing: 0.3px; line-height: 1.2;
}
.top-bar-subtitle {
    font-size: 0.7rem; color: #7a96bb; font-weight: 400;
    letter-spacing: 0.3px; margin-top: 2px;
}
.top-bar-badge {
    margin-left: auto;
    background: rgba(6,150,215,0.12);
    border: 1px solid rgba(6,150,215,0.45);
    color: #5bc8ed;
    font-size: 0.63rem; font-weight: 700;
    padding: 4px 11px;
    border-radius: 5px;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    transition: all 0.22s ease;
}
.top-bar-badge:hover {
    background: rgba(6,150,215,0.24);
    border-color: #0696D7;
    color: #ffffff;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid rgba(13,34,65,0.08);
    border-top: 3px solid #0696D7;
    border-radius: 11px;
    padding: 1.5rem 1.3rem 1.2rem;
    text-align: center;
    box-shadow:
        0 1px 3px rgba(10,22,40,0.04),
        0 4px 12px rgba(10,22,40,0.07),
        0 0 0 1px rgba(10,22,40,0.02);
    transition: transform 0.24s ease, box-shadow 0.24s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 2px 6px rgba(10,22,40,0.05),
        0 12px 28px rgba(10,22,40,0.12),
        0 0 0 1px rgba(10,22,40,0.04);
}
.metric-value {
    font-size: 2.3rem; font-weight: 800;
    color: #0a1628; line-height: 1;
    letter-spacing: -1.5px;
}
.metric-label {
    font-size: 0.65rem; font-weight: 700;
    color: #94a3b8; margin-top: 0.5rem;
    text-transform: uppercase; letter-spacing: 0.9px;
}

/* ── Status pills ── */
.status-ok {
    display: inline-flex; align-items: center; gap: 7px;
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    color: #065f3b; padding: 5px 13px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    border: 1px solid rgba(6,95,59,0.18); letter-spacing: 0.3px;
}
.status-ok::before {
    content: ''; width: 7px; height: 7px;
    background: #10b981; border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.25);
    display: inline-block;
    animation: pulse-green 2.2s infinite;
}
.status-err {
    display: inline-flex; align-items: center; gap: 7px;
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    color: #7f1d1d; padding: 5px 13px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    border: 1px solid rgba(127,29,29,0.18); letter-spacing: 0.3px;
}
.status-err::before {
    content: ''; width: 7px; height: 7px;
    background: #ef4444; border-radius: 50%; display: inline-block;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 2px rgba(16,185,129,0.25); }
    50%       { box-shadow: 0 0 0 5px rgba(16,185,129,0.08); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid rgba(13,34,65,0.08);
    box-shadow: 3px 0 16px rgba(10,22,40,0.06);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #0a1628; font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 0.8rem;
}
.section-label {
    font-size: 0.62rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #f1f5f9;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    border-radius: 7px !important;
    border-color: #e2e8f0 !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stSidebar"] .stNumberInput input:focus,
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #0696D7 !important;
    box-shadow: 0 0 0 3px rgba(6,150,215,0.1) !important;
}
[data-testid="stRadio"] label {
    font-size: 0.84rem !important; color: #374151 !important;
    transition: color 0.15s ease !important;
}
[data-testid="stRadio"] label:hover { color: #0696D7 !important; }

/* ── Primary button (Run Analysis) ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d2550 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(6,150,215,0.35) !important;
    border-radius: 9px !important;
    font-size: 0.83rem !important; font-weight: 700 !important;
    letter-spacing: 0.6px !important; text-transform: uppercase !important;
    padding: 0.68rem 1.2rem !important;
    box-shadow:
        0 4px 16px rgba(10,22,40,0.32),
        0 1px 4px rgba(10,22,40,0.15),
        inset 0 1px 0 rgba(255,255,255,0.07) !important;
    transition: all 0.22s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0d2241 0%, #103070 100%) !important;
    box-shadow:
        0 8px 26px rgba(10,22,40,0.38),
        0 3px 8px rgba(10,22,40,0.2),
        inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(6,150,215,0.55) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.25) !important;
}

/* ── General secondary buttons ── */
.stButton > button {
    font-size: 0.82rem !important; font-weight: 500 !important;
    border-radius: 7px !important;
    transition: all 0.2s ease !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #e8ecf4;
    background: transparent;
    margin-bottom: 1.8rem;
}
[data-baseweb="tab"] {
    font-size: 0.83rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 0.75rem 1.5rem !important;
    color: #64748b !important;
    border-bottom: 2.5px solid transparent !important;
    margin-bottom: -2px !important;
    transition: color 0.2s ease, background 0.2s ease !important;
    background: transparent !important;
}
[data-baseweb="tab"]:hover {
    color: #0696D7 !important;
    background: rgba(6,150,215,0.04) !important;
    border-radius: 6px 6px 0 0 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #0696D7 !important;
    border-bottom: 2.5px solid #0696D7 !important;
    background: rgba(6,150,215,0.04) !important;
    border-radius: 6px 6px 0 0 !important;
}

/* ── Subheader ── */
h3, .stSubheader {
    color: #0a1628 !important; font-weight: 700 !important;
    font-size: 0.97rem !important; letter-spacing: 0.2px !important;
}

/* ── Pro info box ── */
.pro-info {
    background: linear-gradient(135deg, #eff7ff, #e4f0fb);
    border-left: 3px solid #0696D7;
    border-radius: 7px;
    padding: 1rem 1.3rem;
    font-size: 0.85rem; color: #0a1628;
    box-shadow: 0 1px 4px rgba(6,150,215,0.1);
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e4e9f2 !important;
    border-radius: 9px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 6px rgba(10,22,40,0.05) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1.5px solid #0696D7 !important;
    color: #0696D7 !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    border-radius: 7px !important;
    padding: 0.36rem 0.9rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: #0696D7 !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(6,150,215,0.28) !important;
    transform: translateY(-1px) !important;
}

/* ── Expander polish ── */
[data-testid="stExpander"] {
    border: 1px solid #e4e9f2 !important;
    border-radius: 9px !important;
    box-shadow: 0 1px 4px rgba(10,22,40,0.04) !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 14px rgba(10,22,40,0.09) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #374151 !important; padding: 0.8rem 1.1rem !important;
    background: #fafbfc !important;
    transition: background 0.15s ease !important;
}
[data-testid="stExpander"] summary:hover { background: #eff7ff !important; }

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #f1f5f9 !important;
    margin: 1.2rem 0 !important;
}

/* ── Slider accent ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #0696D7 !important;
    border-color: #0696D7 !important;
    box-shadow: 0 0 0 4px rgba(6,150,215,0.18) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, #0696D7, #45c4ec) !important;
    border-radius: 4px !important;
}

/* ── Warnings / errors ── */
[data-testid="stWarningMessage"] {
    background: linear-gradient(135deg, #fffbeb, #fef3c7) !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 7px !important; color: #78350f !important;
}
[data-testid="stErrorMessage"] {
    background: linear-gradient(135deg, #fef2f2, #fee2e2) !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 7px !important; color: #7f1d1d !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important; font-size: 0.78rem !important;
}

/* ── Native Streamlit metrics (tab 3) ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(13,34,65,0.07);
    border-top: 3px solid #0696D7;
    border-radius: 11px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(10,22,40,0.06), 0 1px 3px rgba(10,22,40,0.03);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(10,22,40,0.11), 0 2px 6px rgba(10,22,40,0.06);
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important; font-weight: 800 !important;
    color: #0a1628 !important; letter-spacing: -1px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important; font-weight: 700 !important;
    color: #94a3b8 !important; text-transform: uppercase !important;
    letter-spacing: 0.9px !important;
}

/* ── Chat message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important;
    padding: 0.42rem 0 !important;
    gap: 0.78rem !important; align-items: flex-start !important;
}
[data-testid="stChatMessageContent"] {
    padding: 0.88rem 1.18rem !important;
    max-width: 78% !important;
    font-size: 0.875rem !important; line-height: 1.68 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #0a1628, #0d2550) !important;
    border-radius: 18px 4px 18px 18px !important;
    box-shadow:
        0 4px 16px rgba(10,22,40,0.28),
        0 1px 4px rgba(10,22,40,0.14) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] li {
    color: #e8f0fb !important; margin: 0 !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    border: 1px solid #e4e9f2 !important;
    border-radius: 4px 18px 18px 18px !important;
    box-shadow:
        0 2px 8px rgba(10,22,40,0.07),
        0 1px 3px rgba(10,22,40,0.04) !important;
    color: #0a1628 !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #0696D7, #0579b0) !important;
    border-radius: 50% !important;
    width: 36px !important; height: 36px !important; min-width: 36px !important;
    flex-shrink: 0 !important;
    box-shadow: 0 2px 8px rgba(6,150,215,0.32) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #0a1628, #0d2241) !important;
    border-radius: 50% !important;
    width: 36px !important; height: 36px !important; min-width: 36px !important;
    flex-shrink: 0 !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.28) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1.5px solid #d1dae8 !important;
    background: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.78rem 1.3rem !important;
    resize: none !important;
    transition: border-color 0.22s ease, box-shadow 0.22s ease, background 0.22s ease !important;
    box-shadow: 0 1px 4px rgba(10,22,40,0.05) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #0696D7 !important;
    box-shadow: 0 0 0 3px rgba(6,150,215,0.13), 0 2px 8px rgba(10,22,40,0.07) !important;
    outline: none !important;
    background: #ffffff !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #0696D7, #0579b0) !important;
    border: none !important; border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(6,150,215,0.38) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    background: linear-gradient(135deg, #0579b0, #0462a0) !important;
    transform: scale(1.08) !important;
    box-shadow: 0 4px 14px rgba(6,150,215,0.48) !important;
}

/* ── Welcome / empty state ── */
.chat-welcome {
    text-align: center;
    padding: 2.8rem 1rem 1.6rem;
}
.chat-welcome-orb {
    width: 66px; height: 66px;
    background: linear-gradient(135deg, #0a1628, #0d2550);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1.3rem;
    box-shadow:
        0 8px 28px rgba(10,22,40,0.32),
        0 0 0 8px rgba(10,22,40,0.06),
        0 0 0 16px rgba(10,22,40,0.02);
}
.chat-welcome-title {
    font-size: 1.22rem !important; font-weight: 700 !important;
    color: #0a1628 !important; margin-bottom: 0.42rem !important;
    letter-spacing: -0.3px !important;
}
.chat-welcome-sub {
    font-size: 0.86rem; color: #64748b;
    max-width: 490px; margin: 0 auto 2.2rem;
    line-height: 1.7;
}

/* Chips label */
.chips-label {
    font-size: 0.63rem; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1.1px;
    text-align: center; margin-bottom: 0.8rem;
}

/* Suggested question chip buttons */
[data-testid="baseButton-secondary"] {
    background: #f8fafc !important;
    border: 1.5px solid #e2e9f3 !important;
    border-radius: 9px !important; color: #374151 !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    padding: 0.58rem 0.95rem !important;
    text-align: left !important; white-space: normal !important;
    height: auto !important; min-height: 2.6rem !important;
    line-height: 1.4 !important;
    box-shadow: 0 1px 3px rgba(10,22,40,0.05) !important;
    transition: all 0.22s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, #0696D7, #0579b0) !important;
    border-color: transparent !important; color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(6,150,215,0.32) !important;
}
[data-testid="baseButton-secondary"]:active { transform: translateY(0) !important; }

/* Clear conversation button override */
.clear-btn-row [data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 7px !important; color: #94a3b8 !important;
    font-size: 0.75rem !important;
    padding: 0.3rem 0.82rem !important;
    min-height: 1.9rem !important;
    transform: none !important; box-shadow: none !important;
}
.clear-btn-row [data-testid="baseButton-secondary"]:hover {
    background: #fef2f2 !important;
    border-color: #fecaca !important; color: #dc2626 !important;
    box-shadow: none !important; transform: none !important;
}

/* Typing animation */
.typing-indicator {
    display: flex; align-items: center; gap: 5px; padding: 0.15rem 0;
}
.typing-dot {
    width: 7px; height: 7px;
    background: #0696D7; border-radius: 50%;
    animation: tdot 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes tdot {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
    30% { transform: translateY(-7px); opacity: 1; }
}

/* ── Footer ── */
.pro-footer {
    margin-top: 3.5rem;
    padding: 1.2rem 2rem;
    background: linear-gradient(135deg, #0a1628 0%, #0d2241 100%);
    border-radius: 11px;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 0.5rem;
    box-shadow: 0 4px 20px rgba(10,22,40,0.22);
}
.pro-footer-left {
    font-size: 0.73rem; color: #7a96bb;
    font-weight: 500; letter-spacing: 0.2px;
}
.pro-footer-right {
    font-size: 0.68rem; color: #4a6280; letter-spacing: 0.4px;
}
.pro-footer span { color: #0696D7; font-weight: 600; }
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
        Powered by YOLOv8 · PaddleOCR · NetworkX · Claude AI
    </div>
    <div class="pro-footer-right">
        ISA S5.1 Compliant &nbsp;·&nbsp; ISO 10628 Compliant &nbsp;·&nbsp; 2024
    </div>
</div>
""", unsafe_allow_html=True)
