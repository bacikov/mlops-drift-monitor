"""
Drift Monitor — Live Simulation
streamlit run predictor/simulation_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from sklearn.metrics import mean_absolute_error

from predictor.engine.cleaner  import DataCleaner
from predictor.engine.trainer  import ModelTrainer
from predictor.engine.detector import DriftDetector
from predictor.engine.store    import ModelStore
from predictor.config          import ALGORITHMS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "drift_mlops", "real_data")
SAVED_DIR= os.path.join(BASE_DIR, "saved_sim")
os.makedirs(SAVED_DIR, exist_ok=True)

TARGET = "target"

# Predefined datasets
DATASETS = {
    "Bike Sharing (2011)": {
        "ref":        os.path.join(DATA_DIR, "bike_reference.csv"),
        "stream1":    os.path.join(DATA_DIR, "bike_new_early.csv"),
        "stream2":    os.path.join(DATA_DIR, "bike_new_late.csv"),
        "unit":       "bikes/h",
        "phase1":     "Summer",
        "phase2":     "Autumn",
        "ref_label":  "Spring 2011",
        "stream_start": "2011-06-01",
        "baseline":   63.8,
        "desc":       "Spring → Summer → Autumn. Temperature & humidity drift.",
    },
    "Energy Consumption": {
        "ref":        os.path.join(DATA_DIR, "energy_reference.csv"),
        "stream1":    os.path.join(DATA_DIR, "energy_new_early.csv"),
        "stream2":    os.path.join(DATA_DIR, "energy_new_late.csv"),
        "unit":       "MW",
        "phase1":     "Early",
        "phase2":     "Late",
        "ref_label":  "2002-2015",
        "stream_start": "2016-01-01",
        "baseline":   None,
        "desc":       "2002-2015 reference → 2016-2017 → 2018. Long-term trend drift.",
    },
    "Custom Upload": {
        "ref":        None,
        "stream1":    None,
        "stream2":    None,
        "unit":       "",
        "phase1":     "Stream 1",
        "phase2":     "Stream 2",
        "ref_label":  "Reference",
        "stream_start": "2024-01-01",
        "baseline":   None,
        "desc":       "Upload your own CSV files.",
    },
}

trainer  = ModelTrainer()
detector = DriftDetector()
store    = ModelStore(SAVED_DIR)

MIN_RETRAIN_SAMPLES = 500
COOLDOWN_STEPS      = 6
MEDIUM_ALERT_STEPS  = 8

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Drift Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Base ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #e8e4de;
}
.stApp {
    background: #111014;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Topbar ─────────────────────────────────────────────────── */
.topbar {
    background: #7c1c2e;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    border-bottom: 1px solid #5c1220;
}
.topbar-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: #ffeef2;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding-right: 20px;
    margin-right: 20px;
    border-right: 1px solid rgba(255,238,242,0.2);
}
.topbar-sub {
    font-size: 11px;
    color: rgba(255,238,242,0.55);
    letter-spacing: 0.2px;
}
.topbar-right {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: rgba(255,238,242,0.4);
    letter-spacing: 0.5px;
}

/* ── Control bar ────────────────────────────────────────────── */
.ctrl-bar {
    background: #1a1720;
    border-bottom: 1px solid #2a2730;
    padding: 0 24px;
    height: 44px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Progress bar ───────────────────────────────────────────── */
.prog-wrap {
    background: #1a1720;
    border-bottom: 1px solid #2a2730;
    padding: 0 24px 0;
    height: 3px;
}
.prog-track {
    height: 3px;
    background: #2a2730;
    width: 100%;
}
.prog-fill {
    height: 3px;
    background: #c2415a;
    transition: width 0.3s ease;
}
.stProgress { display: none !important; }

/* ── Stat strip ─────────────────────────────────────────────── */
.stat-strip {
    background: #1a1720;
    border-bottom: 1px solid #2a2730;
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    padding: 0 24px;
}
.stat {
    padding: 12px 16px 12px 0;
    border-right: 1px solid #2a2730;
}
.stat:last-child { border-right: none; }
.stat-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #5a5660;
    margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 22px;
    font-weight: 500;
    color: #e8e4de;
    line-height: 1;
}
.stat-sub {
    font-size: 10px;
    color: #5a5660;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}
.stat-up   { color: #4a9eda; }
.stat-down { color: #c2415a; }
.stat-warn { color: #d4812a; }

/* ── Main content area ──────────────────────────────────────── */
.main-pad {
    padding: 20px 24px;
}

/* ── Section label ───────────────────────────────────────────── */
.section-label {
    font-size: 9px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5a5660;
    border-bottom: 1px solid #2a2730;
    padding-bottom: 6px;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Status pill ─────────────────────────────────────────────── */
.pill {
    display: inline-block;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 6px;
    border-radius: 2px;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pill-ok     { color: #4a9eda; border-color: #4a9eda; background: rgba(74,158,218,.1); }
.pill-warn   { color: #d4812a; border-color: #d4812a; background: rgba(212,129,42,.1); }
.pill-alert  { color: #c2415a; border-color: #c2415a; background: rgba(194,65,90,.1); }
.pill-mute   { color: #5a5660; border-color: #2a2730; background: transparent; }
.pill-future { color: #7a7580; border-color: #3a3740; background: transparent; }

/* ── Num ─────────────────────────────────────────────────────── */
.num {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}

/* ── Event log ───────────────────────────────────────────────── */
.log-wrap {
    background: #0d0c10;
    border: 1px solid #2a2730;
    border-radius: 2px;
    padding: 8px 12px;
    max-height: 480px;
    overflow-y: auto;
}
.log-row {
    display: grid;
    grid-template-columns: 60px 1fr;
    gap: 10px;
    padding: 4px 0;
    border-bottom: 1px solid #1a1720;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    line-height: 1.5;
}
.log-row:last-child { border-bottom: none; }
.log-t { color: #3a3740; }
.log-m { color: #7a7580; }
.log-retrain .log-t { color: #4a9eda; }
.log-retrain .log-m { color: #4a9eda; font-weight: 500; }
.log-alert   .log-t { color: #c2415a; }
.log-alert   .log-m { color: #c2415a; }
.log-warn    .log-t { color: #d4812a; }
.log-warn    .log-m { color: #d4812a; }

/* ── Info panel ──────────────────────────────────────────────── */
.info-panel {
    background: #1a1720;
    border-left: 2px solid #c2415a;
    padding: 10px 14px;
    font-size: 12px;
    color: #9a9598;
    line-height: 1.7;
    margin: 8px 0;
    border-radius: 0 2px 2px 0;
}

/* ── Divider ─────────────────────────────────────────────────── */
.rule { border: none; border-top: 1px solid #2a2730; margin: 16px 0; }

/* ── Sidebar (hidden but styled) ────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1a1720;
    border-right: 1px solid #2a2730;
}
section[data-testid="stSidebar"] label {
    color: #7a7580 !important;
    font-size: 11px !important;
}

/* ── Table ───────────────────────────────────────────────────── */
.stDataFrame { font-size: 11px; }
.stDataFrame td, .stDataFrame th {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    background: #1a1720 !important;
    color: #e8e4de !important;
}
.stDataFrame th { color: #5a5660 !important; }

/* ── Streamlit widget overrides ─────────────────────────────── */
label { color: #7a7580 !important; }

.stButton > button {
    background: #1a1720 !important;
    border: 1px solid #2a2730 !important;
    color: #e8e4de !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 2px !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"] {
    background: #7c1c2e !important;
    border-color: #7c1c2e !important;
    color: #ffeef2 !important;
}
.stButton > button:hover {
    border-color: #c2415a !important;
    color: #c2415a !important;
    background: #1a1720 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #9a2238 !important;
    color: #ffeef2 !important;
    border-color: #9a2238 !important;
}

/* ── Fix Streamlit column container backgrounds ──────────────── */
[data-testid="stHorizontalBlock"] {
    background: #1a1720;
    gap: 0 !important;
    padding: 6px 24px !important;
    border-bottom: 1px solid #2a2730;
}
[data-testid="stHorizontalBlock"] > div {
    padding: 0 !important;
}
/* ── Remove Streamlit's default top gap ─────────────────────── */
[data-testid="stAppViewBlockContainer"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
.element-container { margin: 0 !important; }

.stSelectbox > div > div, .stSlider {
    background: #1a1720 !important;
    border-color: #2a2730 !important;
    color: #e8e4de !important;
    border-radius: 2px !important;
}
.stToggle > label { color: #7a7580 !important; }
.stProgress > div > div { background: #c2415a !important; }

/* ── Text area (log copy) ────────────────────────────────────── */
.stTextArea textarea {
    background: #0d0c10 !important;
    border-color: #2a2730 !important;
    color: #5a5660 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────
PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d0c10",
    font=dict(family="IBM Plex Mono", color="#5a5660", size=9),
    margin=dict(l=40, r=12, t=12, b=32),
    xaxis=dict(gridcolor="#1e1c24", linecolor="#2a2730", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e1c24", linecolor="#2a2730", showgrid=True, zeroline=False),
)

SEV = {
    "none":     ("#4a9eda", "ok"),
    "low":      ("#4a9eda", "ok"),
    "medium":   ("#d4812a", "warn"),
    "high":     ("#c2415a", "alert"),
    "critical": ("#c2415a", "alert"),
}

# ── Helpers ────────────────────────────────────────────────────────
CHART_CFG = {"displayModeBar": False}

def sec(title):
    """Section label."""
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)

def sb_sec(title):
    """Sidebar section header."""
    st.markdown(
        f'<div style="font-size:9px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:1.5px;color:#5a5660;border-bottom:1px solid #2a2730;'
        f'padding-bottom:4px;margin-bottom:8px;margin-top:16px;'
        f'font-family:\'IBM Plex Mono\',monospace;">{title}</div>',
        unsafe_allow_html=True)

def show(fig, height=None):
    """Show plotly chart."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)

def stat_html(label, value, sub="", color="#e8e4de", border=True):
    """Generate stat box HTML."""
    border_style = "border-right:1px solid #2a2730;" if border else ""
    return (f'<div class="stat" style="{border_style}">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{color};">{value}</div>'
            f'<div class="stat-sub">{sub}</div>'
            f'</div>')

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

DEFAULTS = {
    "sim_model": None, "sim_all_results": None, "sim_champion": None,
    "sim_ref_df": None, "sim_stream": None, "sim_stream_idx": 0,
    "sim_phase": "stream1", "sim_history": [], "sim_events": [],
    "sim_running": False, "sim_retrains": 0, "sim_ref_mean": 0.0,
    "sim_feature_cols": [], "step_size": 24, "drift_threshold": 0.5,
    "sim_champion_history": [], "sim_samples_seen": 0,
    "ds_unit": "bikes/h", "ds_phase1": "Summer", "ds_phase2": "Autumn",
    "ds_ref_label": "Spring 2011", "ds_stream_start": "2011-06-01",
    "ds_baseline": 63.8, "ds_name": "Bike Sharing (2011)",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def log(msg, kind="ok"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.sim_events.insert(0, {"ts": ts, "msg": msg, "kind": kind})
    if len(st.session_state.sim_events) > 300:
        st.session_state.sim_events = st.session_state.sim_events[:300]

def do_train(df, label=""):
    cleaned = DataCleaner.clean(df, TARGET)
    fc = [c for c in cleaned.columns if c not in [TARGET, "_task"]]
    X, y = cleaned[fc], cleaned[TARGET]
    results = trainer.train_all(X, y, task="regression")
    champ   = trainer.get_champion(results, metric="r2")
    store.save(results, champ, label)
    return results, champ, fc, float(y.mean())

def predict_batch(batch_df):
    r  = st.session_state.sim_model
    fc = st.session_state.sim_feature_cols
    X  = batch_df[[c for c in fc if c in batch_df.columns]]
    Xi = r.scaler.transform(X) if r.scaler else X.values
    return r.model.predict(Xi)

def do_retrain():
    S      = st.session_state
    seen   = S.sim_stream.iloc[:S.sim_stream_idx]
    seen_gt = seen.dropna(subset=[TARGET]) if TARGET in seen.columns else pd.DataFrame()

    if len(seen_gt) < MIN_RETRAIN_SAMPLES:
        return False, len(seen_gt)

    ref_clean = S.sim_ref_df[[c for c in S.sim_ref_df.columns if c != "_task"]]
    combined  = pd.concat([ref_clean, seen_gt], ignore_index=True)
    cleaned   = DataCleaner.clean(combined, TARGET)
    fc        = [c for c in cleaned.columns if c not in [TARGET, "_task"]]
    X, y      = cleaned[fc], cleaned[TARGET]

    results = trainer.train_all(X, y, task="regression")
    champ   = trainer.get_champion(results, metric="r2")
    r2      = results[champ].metrics.get("r2", "?")
    store.save(results, champ, f"Retrain #{S.sim_retrains+1}")

    S.sim_all_results = results
    S.sim_champion    = champ
    S.sim_model       = results[champ]
    S.sim_retrains   += 1
    S.sim_ref_mean    = float(y.mean())
    S.sim_ref_df      = seen_gt[[c for c in fc if c in seen_gt.columns]].copy()

    S.sim_champion_history.append({
        "retrain": S.sim_retrains,
        "step":    len(S.sim_history) + 1,
        "model":   ALGORITHMS.get(champ, {}).get("label", champ),
        "r2": r2, "samples": len(seen_gt),
    })

    algo = ALGORITHMS.get(champ, {}).get("label", champ)
    log(f"Retrain #{S.sim_retrains} complete · {len(seen_gt):,} samples · R²={r2} · {algo}", "retrain")
    return True, len(seen_gt)

def process_step():
    S    = st.session_state
    stream, idx, step = S.sim_stream, S.sim_stream_idx, S.step_size

    if stream is None or idx >= len(stream):
        return False

    batch = stream.iloc[idx:idx+step].copy()
    S.sim_stream_idx += step

    # Tarih hesapla
    start_date   = date.fromisoformat(S.get("ds_stream_start", "2011-06-01"))
    current_date = start_date + timedelta(hours=int(idx))
    date_str     = current_date.strftime("%a %b %d")

    has_target = TARGET in batch.columns and batch[TARGET].notna().any()
    S.sim_phase = "stream1" if has_target else "stream2"
    PHASE2_LABEL = S.get("ds_phase2", "Stream 2")

    try:
        y_pred    = predict_batch(batch)
        pred_mean = round(float(y_pred.mean()), 1)
    except Exception as e:
        log(f"prediction error — {e}", "alert")
        return True

    actual_mean = mae = None
    if has_target:
        y_true = batch[TARGET].dropna().values
        y_p    = y_pred[:len(y_true)]
        if len(y_true) > 0:
            actual_mean = round(float(y_true.mean()), 1)
            mae         = round(float(mean_absolute_error(y_true, y_p)), 1)
        # O(1) samples counter
        S.sim_samples_seen = S.get("sim_samples_seen", 0) + len(y_true)

    # Drift — cumulative window
    window = stream.iloc[max(0, idx - 200*max(step,1)):idx+step]
    common = [c for c in S.sim_feature_cols
              if c in S.sim_ref_df.columns and c in window.columns]

    drift_score, severity = 0.0, "none"
    if common and len(window) >= 50:
        try:
            dr          = detector.detect(S.sim_ref_df[common], window[common], common)
            drift_score = dr.drift_score
            severity    = dr.severity
        except:
            pass

    step_num     = len(S.sim_history) + 1
    last_retrain = next((h["step"] for h in reversed(S.sim_history)
                         if h.get("event") == "RETRAIN"), 0)
    steps_since  = step_num - last_retrain
    cooldown_ok  = last_retrain == 0 or steps_since >= COOLDOWN_STEPS
    medium_streak = sum(1 for h in S.sim_history[-MEDIUM_ALERT_STEPS:]
                        if h.get("severity") == "medium")
    samples_seen = S.get("sim_samples_seen", 0)

    event, should_retrain, retrain_reason = None, False, ""
    is_high = severity in ["high", "critical"]

    if is_high and has_target and cooldown_ok:
        should_retrain = True
        retrain_reason = f"{severity} drift · {drift_score:.3f}"
    elif medium_streak >= MEDIUM_ALERT_STEPS and has_target and cooldown_ok:
        should_retrain = True
        retrain_reason = f"medium drift · {medium_streak} consecutive steps"
    elif is_high and not cooldown_ok:
        event = "ALERT"
        log(f"step {step_num:03d} · {date_str} · {severity} drift · {drift_score:.3f} · cooldown ({COOLDOWN_STEPS-steps_since} steps)", "warn")
    elif is_high and not has_target:
        event = "ALERT"
        log(f"step {step_num:03d} · {date_str} · {severity} drift {drift_score:.3f} · future phase — no retrain", "warn")

    if should_retrain:
        if samples_seen < MIN_RETRAIN_SAMPLES:
            event = "ALERT"
            act_str = f" · actual={actual_mean}" if actual_mean else ""
            log(f"step {step_num:03d} · {date_str} · {retrain_reason} · accumulating ({samples_seen}/{MIN_RETRAIN_SAMPLES}) · pred={pred_mean}{act_str}", "warn")
        else:
            event = "RETRAIN"
            log(f"step {step_num:03d} · {date_str} · {retrain_reason} · initiating retrain ({samples_seen:,} samples)", "alert")
            with st.spinner("Retraining..."):
                success, n = do_retrain()
            if not success:
                event = "ALERT"

    if event is None:
        act     = f" · actual={actual_mean}" if actual_mean else (f" · {PHASE2_LABEL} (future)" if S.sim_phase == "stream2" else "")
        sev_tag = f" [{severity}]" if severity not in ["none", "low"] else ""
        log(f"step {step_num:03d} · {date_str} · drift={drift_score:.3f}{sev_tag} · pred={pred_mean}{act}", "ok")

    S.sim_history.append({
        "step": step_num, "pred": pred_mean, "actual": actual_mean,
        "mae": mae, "drift_score": round(drift_score, 3),
        "severity": severity, "phase": S.sim_phase,
        "event": event, "idx": idx, "date": date_str,
        "date_full": current_date.isoformat(),
    })
    return True

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    sb_sec("Configuration")

    model_ready = st.session_state.sim_model is not None
    champ       = st.session_state.sim_champion

    sb_sec("Active Model")
    if model_ready and champ:
        r2    = st.session_state.sim_all_results[champ].metrics.get("r2","—")
        label = ALGORITHMS.get(champ,{}).get("label", champ)
        st.markdown(f'<span class="pill pill-ok">active</span>&nbsp;<span class="num" style="font-size:11px;">{label}</span><br><span style="color:#5a5660;font-size:10px;font-family:\'IBM Plex Mono\';">R²&nbsp;{r2}&nbsp;·&nbsp;held-out</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-mute">no model</span>', unsafe_allow_html=True)

    sb_sec("Parameters")
    step_size = st.select_slider("Batch size", options=[1,6,12,24,48],
        value=st.session_state.step_size, format_func=lambda x: f"{x}h")
    st.session_state.step_size = step_size
    drift_thr = st.slider("Retrain threshold", 0.20, 0.80,
        float(st.session_state.drift_threshold), 0.05)
    st.session_state.drift_threshold = drift_thr

    sb_sec("Scenario")
    ds_n = st.session_state.get("ds_name", "Bike Sharing (2011)")
    p1   = st.session_state.get("ds_phase1", "Summer")
    p2   = st.session_state.get("ds_phase2", "Autumn")
    rl   = st.session_state.get("ds_ref_label", "Reference")
    st.markdown(f'<div class="info-panel"><strong>{rl}</strong> — training<br><strong>{p1}</strong> — live validation<br><strong>{p2}</strong> — future prediction</div>', unsafe_allow_html=True)

    if model_ready:
        sb_sec("Phase")
        phase = st.session_state.sim_phase
        if phase == "stream1":
            st.markdown(f'<span class="pill pill-ok">{p1} · ground truth</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-future">{p2} · prediction only</span>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    if model_ready:
        if st.button("↺ Reset", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = ([] if isinstance(v, list) else v)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════════

if model_ready and st.session_state.sim_history:
    _stream = st.session_state.sim_stream
    _total  = len(_stream) if _stream is not None else 1
    _idx    = st.session_state.sim_stream_idx
    _pct    = int(min(_idx / _total, 1) * 100)
    _phase  = st.session_state.sim_phase
    _p1     = st.session_state.get("ds_phase1","Stream 1").upper()
    _p2     = st.session_state.get("ds_phase2","Stream 2").upper()
    _phase_str = f"{_p1} · VALIDATING" if _phase == "stream1" else f"{_p2} · PREDICTING"
    _n_ret  = st.session_state.sim_retrains
    _topbar_right = f"{_idx:,} / {_total:,} rows · {_pct}% · {_phase_str} · {_n_ret} RETRAINS"
elif model_ready:
    _topbar_right = "MODEL ACTIVE · PRESS STEP"
else:
    _topbar_right = "CONFIGURE DATASET → START SIMULATION"

_ds_name = st.session_state.get("ds_name", "Drift Monitor")

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;">
    <span class="topbar-logo">Drift Monitor</span>
    <span class="topbar-sub">{_ds_name} · Automated Drift Detection</span>
  </div>
  <span class="topbar-right">{_topbar_right}</span>
</div>
""", unsafe_allow_html=True)

# ── Progress bar (custom HTML, no st.progress) ─────────────────────
if model_ready and st.session_state.sim_history:
    _pct_w = int(min(st.session_state.sim_stream_idx / max(len(st.session_state.sim_stream), 1), 1) * 100)
    st.markdown(f"""
    <div style="height:3px;background:#2a2730;width:100%;">
      <div style="height:3px;background:#c2415a;width:{_pct_w}%;transition:width 0.3s;"></div>
    </div>
    """, unsafe_allow_html=True)

# ── Controls ───────────────────────────────────────────────────────
st.markdown('<div style="background:#1a1720;border-bottom:1px solid #2a2730;padding:8px 24px;">', unsafe_allow_html=True)

if not model_ready:
    _c1, _c2, _c3 = st.columns([2, 2, 4])
    with _c1:
        ds_name = st.selectbox("Dataset", list(DATASETS.keys()),
                               label_visibility="collapsed",
                               key="ds_selector")
        ds = DATASETS[ds_name]

    with _c2:
        # Custom upload veya predefined
        if ds_name == "Custom Upload":
            ref_file    = st.file_uploader("Reference CSV", type="csv", key="ref_upload")
            stream1_file= st.file_uploader("Stream 1 CSV (ground truth)", type="csv", key="s1_upload")
            stream2_file= st.file_uploader("Stream 2 CSV (future)", type="csv", key="s2_upload")
            unit_inp    = st.text_input("Unit", "units", key="unit_inp")
            p1_inp      = st.text_input("Phase 1 name", "Stream 1", key="p1_inp")
            p2_inp      = st.text_input("Phase 2 name", "Stream 2", key="p2_inp")
            start_inp   = st.text_input("Stream start date", "2024-01-01", key="start_inp")
            has_demo    = ref_file and stream1_file
        else:
            has_demo = all(ds[k] and os.path.exists(ds[k]) for k in ["ref","stream1"])
            ref_file = stream1_file = stream2_file = None
            unit_inp = ds["unit"]; p1_inp = ds["phase1"]; p2_inp = ds["phase2"]
            start_inp = ds["stream_start"]
            if not has_demo:
                st.warning(f"Data files not found for {ds_name}")

    with _c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Start Simulation", type="primary", use_container_width=True, disabled=not has_demo):
            with st.spinner(f"Training on {ds['ref_label'] if ds_name != 'Custom Upload' else 'reference data'}..."):
                # Dosyaları yükle
                if ds_name == "Custom Upload":
                    ref    = pd.read_csv(ref_file)
                    stream1= pd.read_csv(stream1_file)
                    stream2= pd.read_csv(stream2_file).drop(columns=[TARGET], errors="ignore") if stream2_file else pd.DataFrame()
                    baseline = None
                else:
                    ref     = pd.read_csv(ds["ref"])
                    stream1 = pd.read_csv(ds["stream1"])
                    stream2 = pd.read_csv(ds["stream2"]).drop(columns=[TARGET], errors="ignore") if ds["stream2"] else pd.DataFrame()
                    baseline = ds["baseline"]

                results, champ, fc, ref_mean = do_train(
                    ref, ds["ref_label"] if ds_name != "Custom Upload" else "Reference"
                )

                stream = pd.concat([stream1, stream2], ignore_index=True) if not stream2.empty else stream1
                ref_c  = DataCleaner.clean(ref, TARGET)

                st.session_state.update({
                    "sim_all_results": results, "sim_champion": champ,
                    "sim_model": results[champ],
                    "sim_ref_df": ref_c.drop(columns=["_task"], errors="ignore"),
                    "sim_stream": stream, "sim_stream_idx": 0,
                    "sim_feature_cols": fc, "sim_ref_mean": ref_mean,
                    "sim_running": True,
                    # Dataset config
                    "ds_unit":         unit_inp,
                    "ds_phase1":       p1_inp,
                    "ds_phase2":       p2_inp,
                    "ds_ref_label":    ds["ref_label"] if ds_name != "Custom Upload" else "Reference",
                    "ds_stream_start": start_inp,
                    "ds_baseline":     baseline,
                    "ds_name":         ds_name,
                })
                r2 = results[champ].metrics.get("r2","?")
                log(f"Dataset: {ds_name}", "ok")
                log(f"Model trained · {ds['ref_label'] if ds_name != 'Custom Upload' else 'Reference'} · {len(ref):,} rows · {ALGORITHMS.get(champ,{}).get('label',champ)} · R²={r2}", "ok")
                log(f"Stream: {len(stream1):,} rows {p1_inp} (ground truth)" + (f" + {len(stream2):,} rows {p2_inp} (future)" if not stream2.empty else ""), "ok")
            st.rerun()
else:
    stream  = st.session_state.sim_stream
    is_done = stream is None or st.session_state.sim_stream_idx >= len(stream)
    _b1, _b2, _b3, _b4, _b5, _b6, _spacer = st.columns([1, 1, 1, 1, 1, 1, 2])
    with _b1:
        if st.button("Step →", use_container_width=True, disabled=is_done):
            process_step(); st.rerun()
    with _b2:
        auto = st.toggle("Auto", key="_auto")
    with _b3:
        step_size = st.selectbox("Batch", [1,6,12,24,48],
            index=[1,6,12,24,48].index(st.session_state.step_size),
            format_func=lambda x: f"{x}h",
            label_visibility="collapsed")
        st.session_state.step_size = step_size
    with _b4:
        if st.session_state.sim_events:
            log_text = "\n".join([f"{e['ts']} {e['msg']}" for e in reversed(st.session_state.sim_events)])
            st.download_button("↓ Log", data=log_text,
                file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", use_container_width=True)
    with _b5:
        if st.button("↺ Reset", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = ([] if isinstance(v, list) else v)
            st.rerun()
    with _b6:
        n_ret = st.session_state.sim_retrains
        st.markdown(
            f'<div style="padding-top:5px;text-align:center;">'
            f'<span class="num" style="font-size:16px;color:{"#c2415a" if n_ret>0 else "#3a3740"};">{n_ret}</span>'
            f'<span style="font-size:9px;color:#3a3740;text-transform:uppercase;letter-spacing:1px;margin-left:5px;">retrains</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if auto and not is_done:
        import time; process_step(); time.sleep(0.15); st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top:-16px;"></div>', unsafe_allow_html=True)

if not model_ready:
    st.markdown("""
    <div class="info-panel">
    <strong style="color:#e8e4de;">How this works</strong> — A model is trained on Spring 2011 bike rental data (March–May, 2,193 hours).
    Summer 2011 data then streams in, validated against ground truth.
    Drift is detected using KS, PSI and Wasserstein tests.
    When drift exceeds the threshold and sufficient data has accumulated, the model retrains automatically.
    Autumn 2011 follows — no ground truth, pure prediction mode.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

history = st.session_state.sim_history
if not history:
    st.markdown("""
    <div style="padding:60px 0;text-align:center;color:#3a3740;">
        <div style="font-family:'IBM Plex Mono';font-size:32px;margin-bottom:12px;color:#2a2730;">◈</div>
        <div style="font-size:13px;color:#5a5660;">Press <strong style="color:#e8e4de;">Step →</strong> or enable <strong style="color:#e8e4de;">Auto</strong></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# PROGRESS
# ═══════════════════════════════════════════════════════════════════

stream = st.session_state.sim_stream
total  = len(stream) if stream is not None else 1
idx    = st.session_state.sim_stream_idx
_prog_pct = int(min(idx/total, 1.0) * 100)

# ═══════════════════════════════════════════════════════════════════
# STAT ROW
# ═══════════════════════════════════════════════════════════════════

df_h      = pd.DataFrame(history)
latest    = history[-1]
summer_df = df_h[df_h["phase"]=="stream1"]
autumn_df = df_h[df_h["phase"]=="stream2"]
rt_df     = df_h[df_h["event"]=="RETRAIN"]

UNIT     = st.session_state.get("ds_unit", "units")
PHASE1   = st.session_state.get("ds_phase1", "Stream 1")
PHASE2   = st.session_state.get("ds_phase2", "Stream 2")
BASELINE = st.session_state.get("ds_baseline", None)

pred   = latest["pred"]
actual = latest["actual"]
drift  = latest["drift_score"]
sev    = latest["severity"]
sev_color, sev_pill = SEV.get(sev, ("#5a5660", "mute"))
avg_mae = round(float(summer_df["mae"].dropna().mean()), 1) \
          if not summer_df.empty and summer_df["mae"].notna().any() else None
err_pct = round(abs(actual-pred)/max(actual,1)*100, 1) if actual else None
n_ret   = st.session_state.sim_retrains

champ_r2 = champ_label = "—"
if st.session_state.sim_all_results and champ:
    champ_r2    = st.session_state.sim_all_results[champ].metrics.get("r2", "—")
    champ_label = ALGORITHMS.get(champ, {}).get("label", champ)

if actual:
    dc = "stat-up" if actual >= pred else "stat-down"
    actual_html = f'<div class="stat-value">{actual:.0f}</div><div class="stat-sub {dc}">{actual-pred:+.0f} · {err_pct}%</div>'
else:
    actual_html = f'<div class="stat-value" style="color:#5a5660;font-size:12px;padding-top:6px;"><span class="pill pill-future">{PHASE2}</span></div><div class="stat-sub">no ground truth</div>'

if avg_mae:
    vs = round(BASELINE - avg_mae, 1) if BASELINE else None
    mae_html = f'<div class="stat-value">{avg_mae:.1f}</div><div class="stat-sub {"stat-up" if vs and vs>0 else "stat-down" if vs else ""}">{"vs baseline "+f"{vs:+.1f} {UNIT}" if vs else UNIT}</div>'
else:
    mae_html = f'<div class="stat-value" style="color:#5a5660;">—</div><div class="stat-sub">{PHASE1} only</div>'

st.markdown(
    '<div style="background:#1a1720;border-bottom:1px solid #2a2730;'
    'display:grid;grid-template-columns:repeat(7,1fr);padding:0 0 0 24px;">'
    + stat_html("Prediction", f"{pred:.0f}", UNIT)
    + f'<div class="stat"><div class="stat-label">Actual</div>{actual_html}</div>'
    + stat_html("Drift Score", f"{drift:.3f}", f'<span class="pill pill-{sev_pill}">{sev}</span>', sev_color)
    + stat_html("Model R²", str(champ_r2), "held-out test")
    + f'<div class="stat"><div class="stat-label">Avg MAE</div>{mae_html}</div>'
    + stat_html("Retrains", str(n_ret), "auto-triggered", "#c2415a" if n_ret > 0 else "#5a5660")
    + stat_html("Champion", champ_label, f"step {len(history)}", border=False)
    + '</div>',
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════
# CHARTS + LOG
# ═══════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([3,1], gap="small")

with col_left:
    sec("Prediction vs Actual")
    fig = go.Figure()
    ref_mean = st.session_state.sim_ref_mean
    fig.add_hline(y=ref_mean, line_dash="dot", line_color="#2a2730", line_width=1,
                  annotation_text=f"ref {ref_mean:.0f}",
                  annotation_font=dict(size=8, color="#5a5660"),
                  annotation_position="bottom right")
    if not summer_df.empty:
        fig.add_trace(go.Scatter(x=summer_df["step"], y=summer_df["pred"],
            name="predicted", mode="lines", line=dict(color="#4a9eda", width=1.5)))
        av = summer_df["actual"].dropna()
        if not av.empty:
            fig.add_trace(go.Scatter(x=summer_df.loc[av.index,"step"], y=av,
                name="actual", mode="markers",
                marker=dict(color="#e8e4de", size=3.5, symbol="circle")))
    if not autumn_df.empty:
        fig.add_trace(go.Scatter(x=autumn_df["step"], y=autumn_df["pred"],
            name="predicted (future)", mode="lines",
            line=dict(color="#4a9eda", width=1.5, dash="dot")))
    for _, r in rt_df.iterrows():
        fig.add_vline(x=r["step"], line_color="#c2415a", line_width=1, line_dash="dot",
                      annotation_text="retrain",
                      annotation_font=dict(size=8, color="#c2415a"),
                      annotation_position="top left")
    if not summer_df.empty and not autumn_df.empty:
        fig.add_vline(x=summer_df["step"].max(), line_color="#2a2730", line_width=1)
    fig.update_layout(**PLOT, legend=dict(orientation="h", yanchor="bottom", y=1,
        font=dict(size=9), bgcolor="rgba(0,0,0,0)"), xaxis_title="step", yaxis_title=UNIT)
    show(fig, 240)

    dc1, dc2 = st.columns(2)
    with dc1:
        sec("Drift Score")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_h["step"], y=df_h["drift_score"],
            mode="lines", line=dict(color="#7a7580", width=1.5),
            fill="tozeroy", fillcolor="rgba(194,65,90,.08)"))
        thr = st.session_state.drift_threshold
        fig2.add_hline(y=thr, line_color="#c2415a", line_width=1, line_dash="dot")
        for y, c in [(0.15,"#4a9eda"),(0.30,"#d4812a")]:
            fig2.add_hline(y=y, line_color=c, line_width=1, line_dash="dot")
        if not rt_df.empty:
            fig2.add_trace(go.Scatter(x=rt_df["step"], y=rt_df["drift_score"],
                mode="markers", marker=dict(color="#c2415a", size=7, symbol="triangle-up"),
                showlegend=False))
        fig2.update_layout(**PLOT, yaxis_range=[0,1], showlegend=False,
            xaxis_title="step", yaxis_title="score")
        show(fig2, 180)

    with dc2:
        sec("MAE per Step")
        if not summer_df.empty and summer_df["mae"].notna().any():
            mae_v = summer_df["mae"].dropna()
            fig3  = go.Figure()
            fig3.add_trace(go.Bar(
                x=summer_df.loc[mae_v.index,"step"], y=mae_v,
                marker_color=["#4a9eda" if v<20 else "#d4812a" if v<40 else "#c2415a" for v in mae_v],
                marker_opacity=0.8))
            if BASELINE:
                fig3.add_hline(y=BASELINE, line_color="#3a3740", line_width=1, line_dash="dot",
                               annotation_text=f"baseline {BASELINE}",
                               annotation_font=dict(size=8, color="#5a5660"))
            for _, r in rt_df.iterrows():
                fig3.add_vline(x=r["step"], line_color="#c2415a", line_width=1, line_dash="dot")
            fig3.update_layout(**PLOT, showlegend=False, xaxis_title="step", yaxis_title=UNIT)
            show(fig3, 180)
        else:
            st.markdown('<div style="height:180px;display:flex;align-items:center;justify-content:center;color:#5a5660;font-size:11px;">Waiting for ground truth</div>', unsafe_allow_html=True)

    # Retrain Impact + Champion Timeline
    if not rt_df.empty and avg_mae is not None:
        rc1, rc2 = st.columns(2)
        with rc1:
            sec("Retrain Impact — MAE Before vs After")
            labels, before_vals, after_vals = [], [], []
            for i, (_, row) in enumerate(rt_df.iterrows()):
                sr = row["step"]
                b  = df_h[(df_h["step"] < sr) & df_h["mae"].notna()].tail(5)
                a  = df_h[(df_h["step"] > sr) & df_h["mae"].notna()].head(5)
                if not b.empty and not a.empty:
                    labels.append(f"#{i+1} (step {sr})")
                    before_vals.append(round(b["mae"].mean(), 1))
                    after_vals.append(round(a["mae"].mean(), 1))
            if labels:
                fig_ri = go.Figure()
                fig_ri.add_trace(go.Bar(name="Before", x=labels, y=before_vals,
                    marker_color="#c2415a", marker_opacity=0.8))
                fig_ri.add_trace(go.Bar(name="After", x=labels, y=after_vals,
                    marker_color="#4a9eda", marker_opacity=0.8))
                fig_ri.update_layout(**PLOT, barmode="group",
                    legend=dict(orientation="h", yanchor="bottom", y=1,
                                font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                    xaxis_title="", yaxis_title=UNIT)
                show(fig_ri, 180)

        with rc2:
            sec("Champion Timeline")
            init_r2  = st.session_state.sim_all_results[champ].metrics.get("r2","—") \
                       if st.session_state.sim_all_results and champ else "—"
            rows_ct  = [{"#":"init","Step":1,"Champion":champ_label,"R²":init_r2,"N":"—"}]
            rows_ct += [{"#":f"#{h['retrain']}","Step":h["step"],"Champion":h["model"],
                         "R²":h["r2"],"N":f"{h['samples']:,}"}
                        for h in st.session_state.get("sim_champion_history",[])]
            st.dataframe(pd.DataFrame(rows_ct), use_container_width=True, hide_index=True)
            st.caption("Champion = highest R² on held-out test set.")

with col_right:
    sec("Event Log")
    events   = st.session_state.sim_events
    log_text = "\n".join([f"{e['ts']} {e['msg']}" for e in reversed(events)])
    st.download_button("↓ Download log", data=log_text,
        file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain", use_container_width=True)
    st.text_area("", value=log_text, height=48, label_visibility="collapsed", key="log_ta")
    for e in events[:80]:
        cls = {"retrain":"log-retrain","alert":"log-alert","warn":"log-warn"}.get(e["kind"],"")
        st.markdown(f'<div class="log-row {cls}"><span class="log-t">{e["ts"]}</span><span class="log-m">{e["msg"]}</span></div>', unsafe_allow_html=True)

# ── Summary ────────────────────────────────────────────────────────
st.markdown('<hr class="rule">', unsafe_allow_html=True)
sec("Run Summary")
s1, s2, s3 = st.columns(3)
with s1:
    st.dataframe(pd.DataFrame([
        {"Metric":"Total steps",          "Value": len(history)},
        {"Metric":f"{PHASE1} (validated)","Value": len(summer_df)},
        {"Metric":f"{PHASE2} (future)",   "Value": len(autumn_df)},
        {"Metric":"Drift alerts",         "Value": df_h[df_h["event"]=="ALERT"].shape[0]},
        {"Metric":"Auto-retrains",        "Value": n_ret},
    ]), use_container_width=True, hide_index=True)
with s2:
    if avg_mae:
        rows2 = [{"Metric":"Avg MAE","Value":f"{avg_mae:.1f} {UNIT}"}]
        if BASELINE:
            rows2 += [{"Metric":"Baseline","Value":f"{BASELINE} {UNIT}"},
                      {"Metric":"ML vs baseline","Value":f"{BASELINE-avg_mae:+.1f} {UNIT}"}]
        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
with s3:
    if st.session_state.sim_all_results and champ:
        st.dataframe(pd.DataFrame([
            {"Algorithm": ALGORITHMS.get(a,{}).get("label",a) + (" ✓" if a==champ else ""),
             "R²": r.metrics.get("r2","—"),
             "MAE": f"{r.metrics.get('mae','—')} {UNIT}"}
            for a, r in st.session_state.sim_all_results.items()
        ]), use_container_width=True, hide_index=True)