"""
app.py  —  VoxLab: AI Voice Analysis
Dark premium 3-D AI dashboard.
Run with:  streamlit run app.py
"""

import os
import tempfile
import base64
import librosa

import streamlit as st
import pandas as pd

from predict import VoiceAnalyzer
from utils.visualization import (
    plot_waveform, plot_spectrogram, plot_mfcc, plot_confidence_bars,
)
from utils.export_utils import (
    append_history, load_history, export_csv, export_pdf, clear_history,
)

# ───────────────────────────────────────────── page config
st.set_page_config(
    page_title="VoxLab — AI Voice Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────────────────── Dark 3-D premium CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ══════════════════ design tokens ══════════════════ */
:root {
    --bg:          #07090F;
    --bg-2:        #0C1020;
    --surface:     #111827;
    --surface-2:   #1A2236;
    --surface-3:   #243050;
    --primary:     #6366F1;
    --primary-dk:  #4F46E5;
    --primary-lt:  #818CF8;
    --accent:      #06B6D4;
    --violet:      #A78BFA;
    --on-primary:  #FFFFFF;
    --text:        #F1F5F9;
    --text-2:      #CBD5E1;
    --text-muted:  #64748B;
    --border:      rgba(255,255,255,0.07);
    --border-2:    rgba(255,255,255,0.13);
    --border-glow: rgba(99,102,241,0.30);
    --green:       #10B981;
    --yellow:      #F59E0B;
    --red:         #EF4444;

    --r-sm:   6px;
    --r-md:   10px;
    --r-lg:   14px;
    --r-xl:   20px;
    --r-2xl:  28px;
    --r-pill: 999px;

    /* layered 3-D shadows */
    --sh-1: 0 2px 4px rgba(0,0,0,.55), 0 1px 2px rgba(0,0,0,.75);
    --sh-2: 0 4px 12px rgba(0,0,0,.65), 0 2px 4px rgba(0,0,0,.50),
            inset 0 1px 0 rgba(255,255,255,.05);
    --sh-3: 0 8px 24px rgba(0,0,0,.75), 0 4px 8px rgba(0,0,0,.55),
            inset 0 1px 0 rgba(255,255,255,.06);
    --sh-4: 0 16px 48px rgba(0,0,0,.85), 0 8px 16px rgba(0,0,0,.65),
            inset 0 1px 0 rgba(255,255,255,.07);
    --sh-float: 0 24px 64px rgba(0,0,0,.90), 0 12px 24px rgba(0,0,0,.70),
                inset 0 1px 0 rgba(255,255,255,.08);
    --glow-primary: 0 0 24px rgba(99,102,241,.40), 0 0 64px rgba(99,102,241,.15);
    --glow-accent:  0 0 20px rgba(6,182,212,.35),  0 0 48px rgba(6,182,212,.12);
    --glow-green:   0 0 16px rgba(16,185,129,.40),  0 0 40px rgba(16,185,129,.12);
}

/* ══════════════════ base ══════════════════ */
.stApp {
    background:            var(--bg) !important;
    background-image:      radial-gradient(rgba(99,102,241,.055) 1px, transparent 1px) !important;
    background-size:       28px 28px !important;
    font-family:           'Inter', system-ui, sans-serif !important;
    color:                 var(--text) !important;
}
.block-container { padding-top:1.6rem !important; max-width:1200px !important; }
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
h1,h2,h3,h4 { font-family:'Inter',sans-serif; color:var(--text); font-weight:700; }

/* ══════════════════ header / app-bar ══════════════════ */
.appbar {
    display:flex; align-items:center; gap:18px;
    padding:20px 28px 18px;
    background:linear-gradient(135deg,#111827 0%,#1a1f38 55%,#111827 100%);
    border-radius:var(--r-2xl);
    border:1px solid rgba(99,102,241,.22);
    box-shadow: var(--sh-4), var(--glow-primary);
    margin-bottom:26px;
    position:relative; overflow:hidden;
}
/* top shimmer line */
.appbar::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent 0%,
        rgba(99,102,241,.9) 30%,rgba(6,182,212,.9) 70%,transparent 100%);
}
/* bottom reflection */
.appbar::after {
    content:''; position:absolute; bottom:0; left:10%; right:10%; height:1px;
    background:linear-gradient(90deg,transparent,rgba(99,102,241,.12),transparent);
}
.appbar-icon {
    width:58px; height:58px; border-radius:17px; flex-shrink:0;
    background:linear-gradient(135deg,var(--primary) 0%,var(--accent) 100%);
    display:grid; place-items:center; font-size:29px;
    box-shadow:0 6px 22px rgba(99,102,241,.55),inset 0 1px 0 rgba(255,255,255,.22);
}
.appbar-title {
    font-size:25px; font-weight:900; margin:0; line-height:1.15;
    background:linear-gradient(130deg,#fff 25%,rgba(167,139,250,.95) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.appbar-subtitle {
    font-size:12px; color:var(--text-muted); margin-top:3px;
    font-weight:400; letter-spacing:.03em;
}
.appbar-badge {
    margin-left:auto;
    display:inline-flex; align-items:center; gap:8px;
    background:linear-gradient(135deg,rgba(99,102,241,.14),rgba(6,182,212,.09));
    color:var(--primary-lt);
    font-size:11px; font-weight:600; padding:6px 15px;
    border-radius:var(--r-pill);
    border:1px solid rgba(99,102,241,.28);
    font-family:'JetBrains Mono',monospace; white-space:nowrap;
    box-shadow:0 0 14px rgba(99,102,241,.18), inset 0 1px 0 rgba(255,255,255,.06);
}
.appbar-badge::before {
    content:''; width:7px; height:7px; border-radius:50%;
    background:var(--green);
    box-shadow:0 0 7px var(--green),0 0 14px rgba(16,185,129,.5);
    animation:pulsedot 2.2s ease-in-out infinite;
}
@keyframes pulsedot {
    0%,100%{ opacity:1; transform:scale(1); }
    50%    { opacity:.55; transform:scale(.8); }
}

/* ══════════════════ cards ══════════════════ */
.card {
    background:linear-gradient(148deg,var(--surface) 0%,var(--bg-2) 100%);
    border-radius:var(--r-xl);
    border:1px solid var(--border);
    box-shadow:var(--sh-3);
    padding:22px 24px;
    position:relative; overflow:hidden;
    transition:box-shadow .25s ease, border-color .25s ease;
}
/* inner top highlight */
.card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.09),transparent);
}
.card:hover {
    border-color:var(--border-glow);
    box-shadow:var(--sh-4);
}
.card-header {
    display:flex; align-items:center; gap:12px;
    margin-bottom:16px; padding-bottom:14px;
    border-bottom:1px solid var(--border);
}
.card-icon {
    width:40px; height:40px; border-radius:var(--r-md);
    background:linear-gradient(135deg,rgba(99,102,241,.16),rgba(6,182,212,.10));
    border:1px solid rgba(99,102,241,.22);
    display:grid; place-items:center; font-size:20px;
    box-shadow:0 2px 8px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05);
}
.card-title {
    font-size:14px; font-weight:600; color:var(--text-2); letter-spacing:.01em;
}

/* ══════════════════ section label ══════════════════ */
.sec-label {
    display:inline-flex; align-items:center; gap:8px;
    font-family:'JetBrains Mono',monospace;
    font-size:10px; font-weight:600; letter-spacing:.15em;
    text-transform:uppercase; color:var(--primary-lt);
    margin-bottom:14px; opacity:.85;
}
.sec-label::before {
    content:''; display:inline-block;
    width:3px; height:14px; border-radius:2px;
    background:linear-gradient(180deg,var(--primary),var(--accent));
    box-shadow:0 0 8px rgba(99,102,241,.5);
}

/* ══════════════════ metric tiles (3-D lifted) ══════════════════ */
.tile {
    background:linear-gradient(148deg,#161e31 0%,#0f1623 60%,#0c1120 100%);
    border:1px solid var(--border);
    border-radius:var(--r-xl);
    padding:22px 22px 18px;
    position:relative; overflow:hidden;
    box-shadow:
        0 4px 16px rgba(0,0,0,.75),
        0 2px 4px  rgba(0,0,0,.55),
        inset 0  1px 0 rgba(255,255,255,.06),
        inset 0 -1px 0 rgba(0,0,0,.35);
    transform:perspective(700px) translateZ(0);
    transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
    height:100%;
}
.tile::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
}
.tile:hover {
    transform:perspective(700px) translateY(-6px) translateZ(14px);
    border-color:rgba(99,102,241,.28);
    box-shadow:
        0 14px 36px rgba(0,0,0,.85),
        0 5px 10px  rgba(0,0,0,.65),
        inset 0  1px 0 rgba(255,255,255,.09),
        0 0 0 1px rgba(99,102,241,.12),
        0 0 24px   rgba(99,102,241,.06);
}
.tile .t-label {
    font-size:10px; font-weight:600; letter-spacing:.14em;
    text-transform:uppercase; color:var(--text-muted);
    font-family:'JetBrains Mono',monospace; margin-bottom:10px;
}
.tile .t-value {
    font-size:30px; font-weight:800; color:var(--text);
    line-height:1; letter-spacing:-.025em;
}
.tile .t-meta {
    font-size:11px; color:var(--text-muted); margin-top:6px; font-weight:400;
}
.tile.primary {
    border-color:rgba(99,102,241,.26);
    background:linear-gradient(148deg,#1c2240 0%,#151c34 60%,#111827 100%);
    box-shadow:
        0 4px 16px rgba(0,0,0,.75),
        0 2px 4px  rgba(0,0,0,.55),
        inset 0  1px 0 rgba(99,102,241,.18),
        inset 0 -1px 0 rgba(0,0,0,.35),
        0 0 24px   rgba(99,102,241,.07);
}
.tile.primary .t-value {
    background:linear-gradient(135deg,var(--primary-lt) 0%,var(--accent) 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}

/* ══════════════════ confidence chip ══════════════════ */
.chip {
    display:inline-flex; align-items:center; gap:5px;
    font-family:'JetBrains Mono',monospace;
    font-size:10px; font-weight:600; padding:4px 10px;
    border-radius:var(--r-pill); margin-top:8px;
    background:rgba(16,185,129,.10); color:var(--green);
    border:1px solid rgba(16,185,129,.22);
    box-shadow:0 0 9px rgba(16,185,129,.12);
    letter-spacing:.03em;
}
.chip::before { content:"●"; font-size:7px; opacity:.8; }
.chip.low {
    background:rgba(245,158,11,.10); color:#FCD34D;
    border-color:rgba(245,158,11,.22);
    box-shadow:0 0 9px rgba(245,158,11,.12);
}
.chip.low::before { content:"◐"; }

/* ══════════════════ source badge ══════════════════ */
.src-badge {
    display:inline-block; font-size:9px; font-weight:600;
    padding:2px 9px; border-radius:var(--r-pill);
    margin-left:6px; vertical-align:middle;
    font-family:'JetBrains Mono',monospace;
    letter-spacing:.06em; text-transform:uppercase;
}
.src-model     { background:rgba(16,185,129,.10); color:#6EE7B7; border:1px solid rgba(16,185,129,.25); }
.src-heuristic { background:rgba(245,158,11,.10); color:#FDE68A; border:1px solid rgba(245,158,11,.20); }

/* ══════════════════ progress bars ══════════════════ */
.pbar-row { display:flex; align-items:center; gap:10px; margin:7px 0; }
.pbar-label { font-size:11px; color:var(--text-2); width:76px; flex-shrink:0; font-weight:500; }
.pbar-track {
    flex:1; height:5px;
    background:rgba(255,255,255,.06); border-radius:var(--r-pill);
    overflow:hidden; box-shadow:inset 0 1px 2px rgba(0,0,0,.55);
}
.pbar-fill {
    height:100%; border-radius:var(--r-pill);
    background:linear-gradient(90deg,var(--primary),var(--accent));
    box-shadow:0 0 8px rgba(99,102,241,.45);
    transition:width .5s cubic-bezier(.4,0,.2,1);
}
.pbar-pct {
    font-size:10px; color:var(--text-muted); width:32px;
    text-align:right; flex-shrink:0;
    font-family:'JetBrains Mono',monospace;
}

/* ══════════════════ tabs ══════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap:3px !important;
    background:var(--surface) !important;
    border-radius:var(--r-pill) !important;
    padding:4px !important;
    box-shadow:var(--sh-2), inset 0 1px 0 rgba(255,255,255,.04) !important;
    border:1px solid var(--border) !important;
    margin-bottom:22px !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;
    color:var(--text-muted) !important;
    border-radius:var(--r-pill) !important;
    padding:8px 26px !important;
    font-weight:500 !important; font-size:13px !important;
    border:none !important;
    font-family:'Inter',sans-serif !important;
    transition:color .2s !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,var(--primary) 0%,var(--primary-dk) 100%) !important;
    color:#fff !important;
    box-shadow:0 4px 16px rgba(99,102,241,.45),
               inset 0 1px 0 rgba(255,255,255,.18) !important;
}

/* ══════════════════ buttons ══════════════════ */
.stButton > button, .stDownloadButton > button {
    border-radius:var(--r-pill) !important;
    border:1px solid var(--border-2) !important;
    background:var(--surface-2) !important;
    color:var(--text-2) !important;
    font-weight:500 !important; font-size:13px !important;
    padding:9px 22px !important;
    font-family:'Inter',sans-serif !important;
    transition:all .2s ease !important;
    box-shadow:var(--sh-1), inset 0 1px 0 rgba(255,255,255,.05) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color:rgba(99,102,241,.45) !important;
    color:var(--primary-lt) !important;
    background:var(--surface-3) !important;
    box-shadow:var(--sh-2), 0 0 14px rgba(99,102,241,.18) !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,var(--primary) 0%,var(--primary-dk) 100%) !important;
    color:#fff !important; border:none !important;
    box-shadow:0 4px 18px rgba(99,102,241,.55),
               inset 0 1px 0 rgba(255,255,255,.18) !important;
}
.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#7C3AED 0%,var(--primary) 100%) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.65),
               0 0 48px rgba(99,102,241,.22),
               inset 0 1px 0 rgba(255,255,255,.18) !important;
}

/* ══════════════════ file uploader ══════════════════ */
[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,.02) !important;
    border:2px dashed rgba(99,102,241,.25) !important;
    border-radius:var(--r-xl) !important;
    transition:all .2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color:rgba(99,102,241,.55) !important;
    background:rgba(99,102,241,.04) !important;
    box-shadow:0 0 24px rgba(99,102,241,.08) !important;
}

/* ══════════════════ alerts ══════════════════ */
.stAlert { border-radius:var(--r-lg) !important; }

/* ══════════════════ dataframe ══════════════════ */
[data-testid="stDataFrame"] {
    border:1px solid var(--border) !important;
    border-radius:var(--r-lg) !important;
    box-shadow:var(--sh-2) !important;
}

/* ══════════════════ status dots ══════════════════ */
.sdot {
    display:inline-block; width:8px; height:8px;
    border-radius:50%; margin-right:10px; vertical-align:middle; flex-shrink:0;
}
.sdot.on {
    background:var(--green);
    box-shadow:0 0 0 3px rgba(16,185,129,.15), var(--glow-green);
    animation:pulsedot 2.5s ease-in-out infinite;
}
.sdot.off { background:rgba(255,255,255,.12); }
.srow {
    display:flex; align-items:center;
    padding:13px 0; border-bottom:1px solid var(--border);
    font-size:13px; color:var(--text-2);
}
.srow:last-child { border-bottom:none; }
.srow .sname { font-weight:600; color:var(--text); font-size:14px; }
.srow .stag {
    margin-left:auto; font-size:10px;
    font-family:'JetBrains Mono',monospace;
    padding:3px 11px; border-radius:var(--r-pill);
    font-weight:600; letter-spacing:.05em;
}
.stag.live {
    background:rgba(16,185,129,.10); color:#6EE7B7;
    border:1px solid rgba(16,185,129,.25);
    box-shadow:0 0 9px rgba(16,185,129,.18);
}
.stag.idle {
    background:rgba(255,255,255,.04); color:var(--text-muted);
    border:1px solid var(--border);
}

/* ══════════════════ empty state ══════════════════ */
.empty-state {
    text-align:center; padding:72px 28px;
    background:linear-gradient(148deg,var(--surface),var(--bg-2));
    border-radius:var(--r-2xl);
    border:2px dashed rgba(99,102,241,.14);
    color:var(--text-muted); font-size:14px;
    box-shadow:var(--sh-3);
}
.empty-icon {
    font-size:56px; margin-bottom:18px; display:block;
    opacity:.35; filter:drop-shadow(0 0 14px rgba(99,102,241,.35));
}

/* ══════════════════ audio ══════════════════ */
audio { width:100%; border-radius:var(--r-lg); filter:brightness(.9); }

/* ══════════════════ spinner ══════════════════ */
.stSpinner > div { border-top-color:var(--primary) !important; }

/* ══════════════════ scrollbar ══════════════════ */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--surface-3); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--primary); box-shadow:var(--glow-primary); }

/* ══════════════════ responsive ══════════════════ */
@media (max-width:820px) {
    .block-container { padding-left:.6rem !important; padding-right:.6rem !important; }
    .appbar { flex-wrap:wrap; padding:14px 18px; }
    .appbar-title { font-size:21px; }
    .tile .t-value { font-size:24px; }
}
.stApp img, .stApp canvas { max-width:100%; height:auto; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ───────────────────────────────────────────── cache + helpers
@st.cache_resource
def get_analyzer():
    return VoiceAnalyzer()

analyzer = get_analyzer()


def _tile(label, value, meta="", primary=False):
    cls = "tile primary" if primary else "tile"
    meta_html = f'<div class="t-meta">{meta}</div>' if meta else ""
    return (
        f'<div class="{cls}">'
        f'<div class="t-label">{label}</div>'
        f'<div class="t-value">{value}</div>'
        f'{meta_html}'
        f'</div>'
    )


def _chip(conf):
    pct = f"{conf*100:.0f}%"
    cls = "chip low" if conf < 0.60 else "chip"
    return f'<span class="{cls}">{pct} confidence</span>'


def _src(source):
    cls = "src-model" if source == "model" else "src-heuristic"
    label = "AI model" if source == "model" else "heuristic"
    return f'<span class="src-badge {cls}">{label}</span>'


def _pbar(label, pct):
    w = int(pct * 100)
    return (
        f'<div class="pbar-row">'
        f'<span class="pbar-label">{label}</span>'
        f'<div class="pbar-track"><div class="pbar-fill" style="width:{w}%"></div></div>'
        f'<span class="pbar-pct">{w}%</span>'
        f'</div>'
    )


def _distribution_chart(df, column):
    if column not in df.columns or df[column].dropna().empty:
        st.caption("No data yet.")
        return
    counts = (
        df[column].fillna("Unknown").astype(str)
        .value_counts().rename_axis(column).reset_index(name="count")
    )
    st.bar_chart(counts, x=column, y="count", use_container_width=True)


def run_analysis(audio_path, display_name):
    with st.spinner("Analyzing voice signal…"):
        result = analyzer.analyze(audio_path)

    g, a, e = result["gender"], result["age"], result["emotion"]
    p, r    = result["pitch"],  result["speaking_rate"]

    # ── headline tiles ────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Results</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(_tile("Gender", g["value"], primary=True), unsafe_allow_html=True)
    c1.markdown(_chip(g["confidence"]) + _src(g["source"]), unsafe_allow_html=True)
    c2.markdown(_tile("Age Group", a["value"], a.get("range","")), unsafe_allow_html=True)
    c2.markdown(_chip(a["confidence"]) + _src(a["source"]), unsafe_allow_html=True)
    c3.markdown(_tile("Emotion", e["value"]), unsafe_allow_html=True)
    c3.markdown(_chip(e["confidence"]) + _src(e["source"]), unsafe_allow_html=True)

    st.write("")
    c4, c5, c6 = st.columns(3)
    c4.markdown(_tile("Pitch",         f'{p["hz"]} Hz', p["band"] + " range"), unsafe_allow_html=True)
    c5.markdown(_tile("Speaking Rate", r["label"],      f'{r["syllables_per_sec"]} syllables/sec'), unsafe_allow_html=True)
    c6.markdown(_tile("Voice Activity",f'{int(result["speech_ratio"]*100)}%', "of clip is speech"), unsafe_allow_html=True)

    # ── confidence bars ───────────────────────────────────────────────
    st.write("")
    st.markdown('<div class="sec-label">Confidence Breakdown</div>', unsafe_allow_html=True)
    pb1, pb2, pb3 = st.columns(3)

    with pb1:
        st.markdown(
            '<div class="card"><div class="card-title" style="margin-bottom:10px;">Gender</div>'
            + "".join(_pbar(k, v) for k, v in sorted(g["scores"].items(), key=lambda x: -x[1]))
            + "</div>",
            unsafe_allow_html=True,
        )
    with pb2:
        if a["scores"]:
            st.markdown(
                '<div class="card"><div class="card-title" style="margin-bottom:10px;">Age Group</div>'
                + "".join(_pbar(k, v) for k, v in sorted(a["scores"].items(), key=lambda x: -x[1]))
                + "</div>",
                unsafe_allow_html=True,
            )
    with pb3:
        if e["scores"]:
            top_emotions = sorted(e["scores"].items(), key=lambda x: -x[1])[:5]
            st.markdown(
                '<div class="card"><div class="card-title" style="margin-bottom:10px;">Emotion</div>'
                + "".join(_pbar(k, v) for k, v in top_emotions)
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── waveform + spectrograms ───────────────────────────────────────
    st.write("")
    st.markdown('<div class="sec-label">Signal Visualizations</div>', unsafe_allow_html=True)
    signal, sr = result["raw_signal"], result["sr"]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.pyplot(plot_waveform(signal, sr), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(signal, sr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.pyplot(plot_mfcc(signal, sr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── export ────────────────────────────────────────────────────────
    append_history(result, display_name)
    st.write("")
    st.markdown('<div class="sec-label">Export</div>', unsafe_allow_html=True)
    e1, e2, _ = st.columns([1, 1, 2])
    with e1:
        st.download_button(
            "Download CSV",
            data=export_csv(result, display_name),
            file_name=f"{os.path.splitext(display_name)[0]}_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        try:
            pdf_bytes = export_pdf(result, display_name)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{os.path.splitext(display_name)[0]}_analysis.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Install reportlab to enable PDF export.")


# ───────────────────────────────────────────── top app bar
# Detect if real AI models are loaded
_models_loaded = analyzer.gender_model is not None and analyzer.emotion_model is not None
_badge_text    = "AI Models Active" if _models_loaded else "Heuristic Mode"

st.markdown(
    f"""
    <div class="appbar">
        <div class="appbar-icon">🎙️</div>
        <div>
            <div class="appbar-title">VoxLab</div>
            <div class="appbar-subtitle">AI Voice Analysis &nbsp;·&nbsp; Gender &nbsp;·&nbsp; Age &nbsp;·&nbsp; Emotion &nbsp;·&nbsp; Pitch &nbsp;·&nbsp; Rate</div>
        </div>
        <span class="appbar-badge">{_badge_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_analyze, tab_history, tab_about = st.tabs(
    ["  🎙  Analyze  ", "  📋  History  ", "  ℹ  About  "]
)

# ───────────────────────────────────────────── ANALYZE
with tab_analyze:
    st.markdown('<div class="sec-label">Input</div>', unsafe_allow_html=True)

    upload_col, record_col = st.columns(2)

    with upload_col:
        st.markdown(
            '<div class="card">'
            '<div class="card-header">'
            '<div class="card-icon">📁</div>'
            '<div class="card-title">Upload Audio File</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "WAV · MP3 · FLAC · OGG — up to 200 MB",
            type=["wav", "mp3", "flac", "ogg"],
            label_visibility="visible",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with record_col:
        st.markdown(
            '<div class="card">'
            '<div class="card-header">'
            '<div class="card-icon">🎤</div>'
            '<div class="card-title">Record Live Voice</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        recorded = None
        if hasattr(st, "audio_input"):
            recorded = st.audio_input("Tap the mic button to record")
            if recorded is not None:
                # Persist bytes in session state so they survive button-click reruns.
                recorded.seek(0)
                st.session_state["_rec_bytes"] = recorded.read()
            st.caption(
                "Tip: If you see *'An error occurred'*, allow microphone access in your browser "
                "settings and ensure you're accessing via http://localhost."
            )
        else:
            st.info("Upgrade to Streamlit 1.31+ to record directly in the browser.")
        st.markdown('</div>', unsafe_allow_html=True)

    def _audio_suffix(data: bytes) -> str:
        """Detect actual audio format from magic bytes to avoid librosa load errors."""
        if data[:4] == b"RIFF":
            return ".wav"
        if data[:4] == b"OggS":
            return ".ogg"
        if data[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return ".mp3"
        if data[:4] == b"fLaC":
            return ".flac"
        return ".wav"

    audio_bytes, display_name = None, None
    if st.session_state.get("_rec_bytes"):
        audio_bytes, display_name = st.session_state["_rec_bytes"], "recording.wav"
    elif uploaded is not None:
        audio_bytes, display_name = uploaded.getvalue(), uploaded.name

    if audio_bytes:
        st.write("")
        st.audio(audio_bytes)
        if st.button("  Analyze Voice  ", type="primary", use_container_width=False):
            suffix = _audio_suffix(audio_bytes)
            
            # ── voice detection: check if audio is empty or too short ──
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                # Load audio and check duration
                signal, sr = librosa.load(tmp_path, sr=None)
                duration = len(signal) / sr
                
                # Minimum duration check (0.5 seconds)
                if duration < 0.5:
                    st.error("❌ No voice detected — recording is too short. Please record at least 0.5 seconds of audio.")
                else:
                    # Voice detected, proceed with analysis
                    try:
                        run_analysis(tmp_path, display_name)
                    except Exception as exc:
                        st.error(f"Analysis failed — {exc}")
            except Exception as e:
                st.error(f"❌ No voice detected — unable to process audio. Make sure you recorded or uploaded valid audio.")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    else:
        st.markdown(
            '<div class="empty-state">'
            '<span class="empty-icon">🎙️</span>'
            'Upload a voice clip or record live to see the full voice analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

# ───────────────────────────────────────────── HISTORY
with tab_history:
    st.markdown('<div class="sec-label">Session History</div>', unsafe_allow_html=True)
    history = load_history()

    if not history:
        st.markdown(
            '<div class="empty-state">'
            '<span class="empty-icon">📋</span>'
            'No analyses yet — your results will appear here after the first clip.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(history)

        k1, k2, k3 = st.columns(3)
        k1.markdown(_tile("Clips Analyzed", str(len(df)), "this session"), unsafe_allow_html=True)
        top_emotion = df["emotion"].mode().iat[0] if "emotion" in df else "—"
        k2.markdown(_tile("Top Emotion", top_emotion), unsafe_allow_html=True)
        avg_pitch = (
            f'{pd.to_numeric(df["pitch_hz"], errors="coerce").mean():.0f} Hz'
            if "pitch_hz" in df else "—"
        )
        k3.markdown(_tile("Avg Pitch", avg_pitch, "across all clips"), unsafe_allow_html=True)

        st.write("")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.write("")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Gender distribution**")
            _distribution_chart(df, "gender")
        with d2:
            st.markdown("**Emotion distribution**")
            _distribution_chart(df, "emotion")
        with d3:
            st.markdown("**Age group distribution**")
            _distribution_chart(df, "age_group")

        dl_col, clear_col, _ = st.columns([1, 1, 2])
        with dl_col:
            st.download_button(
                "Download full history (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="voxlab_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with clear_col:
            if st.button("Clear History", use_container_width=True):
                clear_history()
                st.success("History cleared.")
                st.rerun()

# ───────────────────────────────────────────── ABOUT
with tab_about:
    st.markdown('<div class="sec-label">About VoxLab</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-header">
                <div class="card-icon">🎙️</div>
                <div class="card-title">VoxLab — AI Voice Analysis</div>
            </div>
            <p style="color:var(--md-text-2); font-size:14px; line-height:1.7; margin:0 0 10px;">
            VoxLab turns a single voice clip into a complete acoustic profile.
            Upload or record audio and it returns gender, age group and emotion —
            each with a confidence score — along with pitch, speaking rate and the
            percentage of the clip containing active speech.
            </p>
            <p style="color:var(--md-text-muted); font-size:13px; margin:0;">
            Models trained on the RAVDESS speech emotion dataset (4 320 samples).
            Gender and emotion use an MLP trained on 20-D MFCC features
            (98.5% gender accuracy, 77.8% emotion accuracy).
            Age uses a pitch-heuristic fallback.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="sec-label">Engine Status</div>', unsafe_allow_html=True)

    status = {
        "Gender model (MLP)":  analyzer.gender_model  is not None,
        "Emotion model (MLP)": analyzer.emotion_model is not None,
        "Age model":           analyzer.age_model     is not None,
        "Pitch & rate analysis":    True,
        "Noise reduction & VAD":    True,
    }
    rows = ""
    for name, ok in status.items():
        dot  = "on" if ok else "off"
        tag  = '<span class="stag live">active</span>' if ok else '<span class="stag idle">standby</span>'
        rows += (
            f'<div class="srow">'
            f'<span class="sdot {dot}"></span>'
            f'<span class="sname">{name}</span>'
            f'{tag}</div>'
        )
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<div style="color:var(--md-hint);font-size:12px;font-family:\'Roboto Mono\',monospace;">'
        'Built with Python &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Librosa &nbsp;·&nbsp; Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )