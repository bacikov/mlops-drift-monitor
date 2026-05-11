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
from datetime import datetime
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
UNIT   = "bikes/h"

SPRING_FILE = os.path.join(DATA_DIR, "bike_reference.csv")
SUMMER_FILE = os.path.join(DATA_DIR, "bike_new_early.csv")
AUTUMN_FILE = os.path.join(DATA_DIR, "bike_new_late.csv")

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

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

DEFAULTS = {
    "sim_model": None, "sim_all_results": None, "sim_champion": None,
    "sim_ref_df": None, "sim_stream": None, "sim_stream_idx": 0,
    "sim_phase": "summer", "sim_history": [], "sim_events": [],
    "sim_running": False, "sim_retrains": 0, "sim_ref_mean": 0.0,
    "sim_feature_cols": [], "step_size": 24, "drift_threshold": 0.5,
    "sim_champion_history": [],
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
    stream = st.session_state.sim_stream
    idx    = st.session_state.sim_stream_idx
    ref_df = st.session_state.sim_ref_df

    seen    = stream.iloc[:idx]
    seen_gt = seen.dropna(subset=[TARGET]) if TARGET in seen.columns else pd.DataFrame()

    if len(seen_gt) < MIN_RETRAIN_SAMPLES:
        return False, len(seen_gt)

    combined = pd.concat([
        ref_df[[c for c in ref_df.columns if c != "_task"]],
        seen_gt,
    ], ignore_index=True)

    cleaned = DataCleaner.clean(combined, TARGET)
    fc = [c for c in cleaned.columns if c not in [TARGET, "_task"]]
    X, y    = cleaned[fc], cleaned[TARGET]
    results = trainer.train_all(X, y, task="regression")
    champ   = trainer.get_champion(results, metric="r2")
    r2      = results[champ].metrics.get("r2", "?")
    store.save(results, champ, f"Retrain #{st.session_state.sim_retrains+1}")

    st.session_state.sim_all_results = results
    st.session_state.sim_champion    = champ
    st.session_state.sim_model       = results[champ]
    st.session_state.sim_retrains   += 1
    st.session_state.sim_ref_mean    = float(y.mean())

    # Champion geçmişini kaydet
    if "sim_champion_history" not in st.session_state:
        st.session_state.sim_champion_history = []
    st.session_state.sim_champion_history.append({
        "retrain": st.session_state.sim_retrains,
        "step":    len(st.session_state.sim_history) + 1,
        "model":   ALGORITHMS.get(champ, {}).get("label", champ),
        "r2":      r2,
        "samples": len(seen_gt),
    })

    fc_ = st.session_state.sim_feature_cols
    avail = [c for c in fc_ if c in seen_gt.columns]
    if avail:
        st.session_state.sim_ref_df = seen_gt[avail].copy()

    log(
        f"Retrain #{st.session_state.sim_retrains} complete · "
        f"{len(seen_gt):,} samples · "
        f"R²={r2} (held-out) · "
        f"{ALGORITHMS.get(champ,{}).get('label',champ)}",
        "retrain"
    )
    return True, len(seen_gt)

def process_step():
    stream = st.session_state.sim_stream
    idx    = st.session_state.sim_stream_idx
    step   = st.session_state.step_size

    if stream is None or idx >= len(stream):
        return False

    batch = stream.iloc[idx:idx+step].copy()
    st.session_state.sim_stream_idx += step

    # Tarih hesapla — yaz Haziran 1 2011'den başlıyor
    from datetime import date, timedelta
    STREAM_START = date(2011, 6, 1)
    current_date = STREAM_START + timedelta(hours=int(idx))
    date_str     = current_date.strftime("%a %b %d")

    has_target = TARGET in batch.columns and batch[TARGET].notna().any()
    st.session_state.sim_phase = "summer" if has_target else "autumn"

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

    # Drift — cumulative window
    ref_df = st.session_state.sim_ref_df
    fc     = st.session_state.sim_feature_cols
    window = stream.iloc[max(0, idx - 200*max(step,1)):idx+step]
    common = [c for c in fc if c in ref_df.columns and c in window.columns]

    drift_score, severity = 0.0, "none"
    if common and len(window) >= 50:
        try:
            dr          = detector.detect(ref_df[common], window[common], common)
            drift_score = dr.drift_score
            severity    = dr.severity
        except:
            pass

    step_num = len(st.session_state.sim_history) + 1
    last_retrain = next(
        (h["step"] for h in reversed(st.session_state.sim_history)
         if h.get("event") == "RETRAIN"), 0
    )
    steps_since   = step_num - last_retrain
    cooldown_ok   = steps_since >= COOLDOWN_STEPS
    recent        = st.session_state.sim_history[-MEDIUM_ALERT_STEPS:]
    medium_streak = sum(1 for h in recent if h.get("severity") == "medium")
    samples_seen  = len(
        stream.iloc[:idx+step].dropna(subset=[TARGET])
        if TARGET in stream.columns else stream.iloc[:idx+step]
    )

    event = None
    should_retrain = False
    retrain_reason = ""

    if severity in ["high","critical"] and has_target and cooldown_ok:
        should_retrain = True
        retrain_reason = f"{severity} drift · {drift_score:.3f}"
    elif medium_streak >= MEDIUM_ALERT_STEPS and has_target and cooldown_ok:
        should_retrain = True
        retrain_reason = f"medium drift · {medium_streak} consecutive steps"
    elif severity in ["high","critical"] and not cooldown_ok:
        event = "ALERT"
        if step_num % 3 == 0:
            log(f"step {step_num:03d} · {date_str} · {severity} drift {drift_score:.3f} · cooldown ({COOLDOWN_STEPS-steps_since} steps)", "warn")
    elif severity in ["high","critical"] and not has_target:
        event = "ALERT"
        log(f"step {step_num:03d} · {date_str} · {severity} drift {drift_score:.3f} · future phase — no retrain", "warn")

    if should_retrain:
        if samples_seen < MIN_RETRAIN_SAMPLES:
            event = "ALERT"
            if step_num % 5 == 0:
                log(f"step {step_num:03d} · {date_str} · {retrain_reason} · accumulating ({samples_seen}/{MIN_RETRAIN_SAMPLES})", "warn")
        else:
            event = "RETRAIN"
            log(f"step {step_num:03d} · {date_str} · {retrain_reason} · initiating retrain ({samples_seen:,} samples)", "alert")
            with st.spinner("Retraining..."):
                success, n = do_retrain()
            if not success:
                event = "ALERT"

    if event is None:
        act  = f" · actual={actual_mean}" if actual_mean else (" · future" if st.session_state.sim_phase == "autumn" else "")
        sev_tag = f" [{severity}]" if severity not in ["none","low"] else ""
        log(f"step {step_num:03d} · {date_str} · drift={drift_score:.3f}{sev_tag} · pred={pred_mean}{act}", "ok")

    st.session_state.sim_history.append({
        "step": step_num, "pred": pred_mean, "actual": actual_mean,
        "mae": mae, "drift_score": round(drift_score,3),
        "severity": severity, "phase": st.session_state.sim_phase,
        "event": event, "idx": idx, "date": date_str,
        "date_full": current_date.isoformat(),
    })
    return True

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="font-size:9px;font-weight:500;text-transform:uppercase;
      letter-spacing:1.5px;color:#c2415a;border-bottom:1px solid #2a2730;
      padding-bottom:6px;margin-bottom:14px;font-family:'IBM Plex Mono',monospace;">
      Configuration</div>
    """, unsafe_allow_html=True)

    model_ready = st.session_state.sim_model is not None
    champ       = st.session_state.sim_champion

    st.markdown("""
    <div style="font-size:9px;font-weight:500;text-transform:uppercase;
      letter-spacing:1.5px;color:#5a5660;border-bottom:1px solid #2a2730;
      padding-bottom:4px;margin-bottom:8px;font-family:'IBM Plex Mono',monospace;">
      Active Model</div>
    """, unsafe_allow_html=True)
    if model_ready and champ:
        r2    = st.session_state.sim_all_results[champ].metrics.get("r2","—")
        label = ALGORITHMS.get(champ,{}).get("label", champ)
        st.markdown(f"""
        <span class="pill pill-ok">active</span>&nbsp;
        <span class="num" style="font-size:11px;">{label}</span><br>
        <span style="color:#5a5660;font-size:10px;font-family:'IBM Plex Mono';">
        R²&nbsp;{r2}&nbsp;·&nbsp;held-out test</span>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-mute">no model</span>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:9px;font-weight:500;text-transform:uppercase;
      letter-spacing:1.5px;color:#5a5660;border-bottom:1px solid #2a2730;
      padding-bottom:4px;margin-bottom:8px;margin-top:16px;font-family:'IBM Plex Mono',monospace;">
      Parameters</div>
    """  , unsafe_allow_html=True)
    step_size = st.select_slider("Batch size",
        options=[1,6,12,24,48], value=st.session_state.step_size,
        format_func=lambda x: f"{x}h")
    st.session_state.step_size = step_size

    drift_thr = st.slider("Retrain threshold",
        0.20, 0.80, float(st.session_state.drift_threshold), 0.05)
    st.session_state.drift_threshold = drift_thr

    st.markdown("""
    <div style="font-size:9px;font-weight:500;text-transform:uppercase;
      letter-spacing:1.5px;color:#5a5660;border-bottom:1px solid #2a2730;
      padding-bottom:4px;margin-bottom:8px;margin-top:16px;font-family:'IBM Plex Mono',monospace;">
      Scenario</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
    <strong>Spring 2011</strong> — initial training<br>
    <strong>Summer 2011</strong> — live validation<br>
    <strong>Autumn 2011</strong> — future prediction
    </div>
    """, unsafe_allow_html=True)

    if model_ready:
        st.markdown("""
        <div style="font-size:9px;font-weight:500;text-transform:uppercase;
          letter-spacing:1.5px;color:#5a5660;border-bottom:1px solid #2a2730;
          padding-bottom:4px;margin-bottom:8px;margin-top:16px;font-family:'IBM Plex Mono',monospace;">
          Phase</div>
        """, unsafe_allow_html=True)
        phase = st.session_state.sim_phase
        if phase == "summer":
            st.markdown('<span class="pill pill-ok">summer · ground truth</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-future">autumn · prediction only</span>', unsafe_allow_html=True)

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
    _phase_str = "SUMMER · VALIDATING" if _phase == "summer" else "AUTUMN · PREDICTING"
    _n_ret  = st.session_state.sim_retrains
    _topbar_right = f"{_idx:,} / {_total:,} h · {_pct}% · {_phase_str} · {_n_ret} RETRAINS"
elif model_ready:
    _topbar_right = "MODEL ACTIVE · PRESS STEP"
else:
    _topbar_right = "SPRING → SUMMER → AUTUMN · 2011"

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;">
    <span class="topbar-logo">Drift Monitor</span>
    <span class="topbar-sub">Bike Demand Forecasting · Automated Drift Detection</span>
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
    has_demo = all(os.path.exists(f) for f in [SPRING_FILE, SUMMER_FILE, AUTUMN_FILE])
    _c1, _c2 = st.columns([2, 6])
    with _c1:
        if st.button("▶  Start Simulation", type="primary", use_container_width=True, disabled=not has_demo):
            with st.spinner("Training on Spring 2011..."):
                ref = pd.read_csv(SPRING_FILE)
                results, champ, fc, ref_mean = do_train(ref, "Spring 2011")
                summer = pd.read_csv(SUMMER_FILE)
                autumn = pd.read_csv(AUTUMN_FILE).drop(columns=[TARGET], errors="ignore")
                stream = pd.concat([summer, autumn], ignore_index=True)
                ref_c  = DataCleaner.clean(ref, TARGET)
                st.session_state.update({
                    "sim_all_results": results, "sim_champion": champ,
                    "sim_model": results[champ],
                    "sim_ref_df": ref_c.drop(columns=["_task"], errors="ignore"),
                    "sim_stream": stream, "sim_stream_idx": 0,
                    "sim_feature_cols": fc, "sim_ref_mean": ref_mean,
                    "sim_running": True,
                })
                r2 = results[champ].metrics.get("r2","?")
                log(f"Model trained · Spring 2011 · {len(ref):,} h · {ALGORITHMS.get(champ,{}).get('label',champ)} · R²={r2}", "ok")
                log(f"Stream: {len(summer):,}h summer (ground truth) + {len(autumn):,}h autumn (future)", "ok")
            st.rerun()
    if not has_demo:
        st.error("Run prepare_bikesharing.py first")
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

# ═══════════════════════════════════════════════════════════════════
# SETUP / EMPTY STATES
# ═══════════════════════════════════════════════════════════════════

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

df_h    = pd.DataFrame(history)
latest  = history[-1]
summer_df = df_h[df_h["phase"]=="summer"]
autumn_df = df_h[df_h["phase"]=="autumn"]

pred   = latest["pred"]
actual = latest["actual"]
drift  = latest["drift_score"]
sev    = latest["severity"]
sev_color, sev_pill = SEV.get(sev, ("#5a5660","mute"))

avg_mae = round(float(summer_df["mae"].dropna().mean()), 1) \
          if not summer_df.empty and summer_df["mae"].notna().any() else None

err_pct = round(abs(actual-pred)/max(actual,1)*100,1) if actual else None

# ── Stat row — pure HTML, no st.columns ───────────────────────────
ref_avg  = round(st.session_state.sim_ref_mean, 0)
n_ret    = st.session_state.sim_retrains
phase    = latest["phase"]
p_label  = "summer" if phase == "summer" else "autumn"
p_pill   = "ok" if phase == "summer" else "future"

# Model R2 for stat row
champ_r2    = "—"
champ_label = "—"
if st.session_state.sim_all_results and champ:
    champ_r2    = st.session_state.sim_all_results[champ].metrics.get("r2", "—")
    champ_label = ALGORITHMS.get(champ, {}).get("label", champ)

if actual:
    delta_val = f"+{actual-pred:.0f}" if actual >= pred else f"{actual-pred:.0f}"
    actual_html = f'<div class="stat-value">{actual:.0f}</div><div class="stat-sub {"stat-up" if actual>=pred else "stat-down"}">{delta_val} · {err_pct}%</div>'
else:
    actual_html = '<div class="stat-value" style="color:#5a5660;">—</div><div class="stat-sub">future</div>'

if avg_mae:
    vs_pers = round(63.8 - avg_mae, 1)
    mae_html = f'<div class="stat-value">{avg_mae:.1f}</div><div class="stat-sub {"stat-up" if vs_pers>0 else "stat-down"}">vs persistence {vs_pers:+.1f} {UNIT}</div>'
else:
    mae_html = '<div class="stat-value" style="color:#5a5660;">—</div><div class="stat-sub">summer only</div>'

st.markdown(f"""
<div style="background:#1a1720;border-bottom:1px solid #2a2730;
  display:grid;grid-template-columns:repeat(7,1fr);padding:0 0 0 24px;">

  <div class="stat">
    <div class="stat-label">Prediction</div>
    <div class="stat-value">{pred:.0f}</div>
    <div class="stat-sub">{UNIT}</div>
  </div>

  <div class="stat">
    <div class="stat-label">Actual</div>
    {actual_html}
  </div>

  <div class="stat">
    <div class="stat-label">Drift Score</div>
    <div class="stat-value" style="color:{sev_color};">{drift:.3f}</div>
    <div class="stat-sub"><span class="pill pill-{sev_pill}">{sev}</span></div>
  </div>

  <div class="stat">
    <div class="stat-label">Model R²</div>
    <div class="stat-value">{champ_r2}</div>
    <div class="stat-sub">held-out test</div>
  </div>

  <div class="stat">
    <div class="stat-label">Avg MAE</div>
    {mae_html}
  </div>

  <div class="stat">
    <div class="stat-label">Retrains</div>
    <div class="stat-value" style="color:{'#c2415a' if n_ret>0 else '#5a5660'};">{n_ret}</div>
    <div class="stat-sub">auto-triggered</div>
  </div>

  <div class="stat" style="border-right:none;">
    <div class="stat-label">Champion</div>
    <div class="stat-value" style="font-size:13px;padding-top:4px;">{champ_label}</div>
    <div class="stat-sub">step {len(history)}</div>
  </div>

</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CHARTS + LOG
# ═══════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([3,1], gap="small")

with col_left:

    # X ekseni için tarih sütunu
    x_axis = df_h["date_full"] if "date_full" in df_h.columns else df_h["step"]
    x_summer = summer_df["date_full"] if "date_full" in summer_df.columns else summer_df["step"]
    x_autumn = autumn_df["date_full"] if "date_full" in autumn_df.columns else autumn_df["step"]
    x_title  = "date" if "date_full" in df_h.columns else "step"

    # Prediction chart
    st.markdown('<div class="section-label">Prediction vs Actual</div>', unsafe_allow_html=True)

    fig = go.Figure()
    ref_mean = st.session_state.sim_ref_mean

    fig.add_hline(y=ref_mean, line_dash="dot", line_color="#2a2730", line_width=1,
                  annotation_text=f"ref {ref_mean:.0f}",
                  annotation_font=dict(size=8,color="#5a5660"),
                  annotation_position="bottom right")

    if not summer_df.empty:
        fig.add_trace(go.Scatter(
            x=summer_df["step"], y=summer_df["pred"],
            name="predicted", mode="lines",
            line=dict(color="#4a9eda", width=1.5),
        ))
        av = summer_df["actual"].dropna()
        if not av.empty:
            fig.add_trace(go.Scatter(
                x=summer_df.loc[av.index,"step"], y=av,
                name="actual", mode="markers",
                marker=dict(color="#e8e4de", size=3.5, symbol="circle"),
            ))

    if not autumn_df.empty:
        fig.add_trace(go.Scatter(
            x=autumn_df["step"], y=autumn_df["pred"],
            name="predicted (future)", mode="lines",
            line=dict(color="#4a9eda", width=1.5, dash="dot"),
        ))

    for _, r in df_h[df_h["event"]=="RETRAIN"].iterrows():
        fig.add_vline(x=r["step"], line_color="#c2415a", line_width=1, line_dash="dot",
                      annotation_text="retrain",
                      annotation_font=dict(size=8,color="#c2415a"),
                      annotation_position="top left")

    if not summer_df.empty and not autumn_df.empty:
        fig.add_vline(x=summer_df["step"].max(), line_color="#2a2730", line_width=1)

    fig.update_layout(**PLOT, height=240,
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        xaxis_title="step", yaxis_title=UNIT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # Drift + MAE
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown('<div class="section-label">Drift Score</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_h["step"], y=df_h["drift_score"],
            mode="lines", name="drift score",
            line=dict(color="#7a7580", width=1.5),
            fill="tozeroy", fillcolor="rgba(194,65,90,.08)",
        ))
        fig2.add_hline(y=st.session_state.drift_threshold,
                       line_color="#c2415a", line_width=1, line_dash="dot",
                       annotation_text="threshold",
                       annotation_font=dict(size=8,color="#c2415a"),
                       annotation_position="bottom right")
        for y,c,l in [(0.15,"#4a9eda","low"),(0.30,"#d4812a","medium")]:
            fig2.add_hline(y=y, line_color=c, line_width=1, line_dash="dot",
                           annotation_text=l,
                           annotation_font=dict(size=8,color=c),
                           annotation_position="bottom right")
        rt = df_h[df_h["event"]=="RETRAIN"]
        if not rt.empty:
            fig2.add_trace(go.Scatter(
                x=rt["step"], y=rt["drift_score"], mode="markers",
                marker=dict(color="#c2415a", size=7, symbol="triangle-up"),
                showlegend=False, name="retrain",
            ))
        fig2.update_layout(**PLOT, height=180, yaxis_range=[0,1],
            showlegend=False,
            xaxis_title="step", yaxis_title="score")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    with dc2:
        st.markdown('<div class="section-label">MAE per Step</div>', unsafe_allow_html=True)
        if not summer_df.empty and summer_df["mae"].notna().any():
            mae_v = summer_df["mae"].dropna()
            fig3  = go.Figure()
            fig3.add_trace(go.Bar(
                x=summer_df.loc[mae_v.index,"step"], y=mae_v,
                marker_color=["#4a9eda" if v<20 else "#d4812a" if v<40 else "#c2415a" for v in mae_v],
                marker_opacity=0.8,
            ))
            fig3.add_hline(y=63.8, line_color="#3a3740", line_width=1, line_dash="dot",
                           annotation_text="persistence baseline 63.8",
                           annotation_font=dict(size=8,color="#5a5660"))
            for _,r in df_h[df_h["event"]=="RETRAIN"].iterrows():
                fig3.add_vline(x=r["step"], line_color="#c2415a", line_width=1, line_dash="dot")
            fig3.update_layout(**PLOT, height=180, showlegend=False,
                xaxis_title="step", yaxis_title=UNIT)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown('<div style="height:180px;display:flex;align-items:center;justify-content:center;color:#5a5660;font-size:11px;">Waiting for ground truth</div>', unsafe_allow_html=True)

    # Retrain Impact + Champion Timeline
    rt_df = df_h[df_h["event"]=="RETRAIN"]
    if not rt_df.empty and avg_mae is not None:
        rc1, rc2 = st.columns(2)

        with rc1:
            st.markdown('<div class="section-label">Retrain Impact — MAE Before vs After</div>', unsafe_allow_html=True)
            labels, before_vals, after_vals = [], [], []
            for i, (_, row) in enumerate(rt_df.iterrows()):
                step_r = row["step"]
                label  = f"Retrain #{i+1} (step {step_r})"
                # MAE before: ortalama son 5 step retrain'den önce
                before_steps = df_h[(df_h["step"] < step_r) & (df_h["mae"].notna())].tail(5)
                # MAE after: retrain'den sonraki ilk 5 step
                after_steps  = df_h[(df_h["step"] > step_r) & (df_h["mae"].notna())].head(5)
                if not before_steps.empty and not after_steps.empty:
                    labels.append(label)
                    before_vals.append(round(before_steps["mae"].mean(), 1))
                    after_vals.append(round(after_steps["mae"].mean(), 1))

            if labels:
                fig_ri = go.Figure()
                fig_ri.add_trace(go.Bar(
                    name="Before retrain", x=labels, y=before_vals,
                    marker_color="#c2415a", marker_opacity=0.8,
                ))
                fig_ri.add_trace(go.Bar(
                    name="After retrain", x=labels, y=after_vals,
                    marker_color="#4a9eda", marker_opacity=0.8,
                ))
                fig_ri.update_layout(**PLOT, height=180,
                    barmode="group",
                    legend=dict(orientation="h", yanchor="bottom", y=1,
                                font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                    xaxis_title="", yaxis_title=UNIT,
                )
                st.plotly_chart(fig_ri, use_container_width=True, config={"displayModeBar":False})

        with rc2:
            st.markdown('<div class="section-label">Champion Timeline</div>', unsafe_allow_html=True)
            champ_hist = st.session_state.get("sim_champion_history", [])

            # İlk model — simülasyon başlangıcı
            initial_r2    = st.session_state.sim_all_results[champ].metrics.get("r2","—") \
                            if st.session_state.sim_all_results and champ else "—"
            initial_label = ALGORITHMS.get(champ,{}).get("label", champ) if champ else "—"

            rows_ct = [{"#": "init", "Step": 1,
                        "Champion": initial_label,
                        "R² (held-out)": initial_r2,
                        "Samples": "2,193"}]

            for h in champ_hist:
                rows_ct.append({
                    "#":             f"#{h['retrain']}",
                    "Step":          h["step"],
                    "Champion":      h["model"],
                    "R² (held-out)": h["r2"],
                    "Samples":       f"{h['samples']:,}",
                })

            st.dataframe(pd.DataFrame(rows_ct), use_container_width=True, hide_index=True)
            st.caption("Champion = highest R² on held-out test set after each retrain.")

with col_right:
    st.markdown('<div class="section-label">Event Log</div>', unsafe_allow_html=True)

    events   = st.session_state.sim_events
    log_text = "\n".join([f"{e['ts']} {e['msg']}" for e in reversed(events)])

    # Copy + download yan yana
    _lc1, _lc2 = st.columns(2)
    with _lc1:
        st.download_button("↓ Download", data=log_text,
            file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain", use_container_width=True)
    with _lc2:
        st.text_area("📋 Copy", value=log_text, height=34,
                     label_visibility="collapsed", key="log_ta")

    st.markdown('<div class="log-wrap">', unsafe_allow_html=True)
    for e in events[:80]:
        cls = {"retrain":"log-retrain","alert":"log-alert","warn":"log-warn"}.get(e["kind"],"")
        st.markdown(
            f'<div class="log-row {cls}">'
            f'<span class="log-t">{e["ts"]}</span>'
            f'<span class="log-m">{e["msg"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Run Summary</div>', unsafe_allow_html=True)

s1,s2,s3 = st.columns(3)
with s1:
    rows = [
        {"":  "Total steps",               "Value": len(history)},
        {"":  "Summer (validated)",         "Value": len(summer_df)},
        {"":  "Autumn (future)",            "Value": len(autumn_df)},
        {"":  "Drift alerts",               "Value": df_h[df_h["event"]=="ALERT"].shape[0]},
        {"":  "Auto-retrains",              "Value": st.session_state.sim_retrains},
    ]
    st.dataframe(pd.DataFrame(rows).rename(columns={"":"Metric"}),
                 use_container_width=True, hide_index=True)

with s2:
    if avg_mae:
        rows2 = [
            {"":  "Avg MAE (summer)",         "Value": f"{avg_mae:.1f} {UNIT}"},
            {"":  "Persistence baseline",     "Value": f"63.8 {UNIT}"},
            {"":  "Mean baseline",            "Value": f"124.0 {UNIT}"},
            {"":  "ML vs persistence",        "Value": f"{63.8-avg_mae:+.1f} {UNIT}"},
        ]
        st.dataframe(pd.DataFrame(rows2).rename(columns={"":"Metric"}),
                     use_container_width=True, hide_index=True)

with s3:
    if st.session_state.sim_all_results and champ:
        rows3 = []
        for a, r in st.session_state.sim_all_results.items():
            rows3.append({
                "Algorithm": ALGORITHMS.get(a,{}).get("label",a) + (" ✓" if a==champ else ""),
                "R²":  r.metrics.get("r2","—"),
                "MAE": f"{r.metrics.get('mae','—')} {UNIT}",
            })
        st.dataframe(pd.DataFrame(rows3), use_container_width=True, hide_index=True)