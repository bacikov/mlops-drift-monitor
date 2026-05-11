"""
Drift Monitor — MLOps Dashboard
================================
Real-time drift detection with live simulation demo.

Tabs:
  🚀 Live Demo  — Watch drift happen in real-time with airline data
  📊 Analysis   — Upload your own data and analyze
  📈 History    — Track all analyses over time
  💾 Models     — Manage saved model versions

Usage:
    streamlit run drift_mlops/monitor.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, pickle, time, urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from drift_mlops.detection.statistical import StatisticalDriftDetector
from drift_mlops.detection.scorer import DriftScorer
from drift_mlops.config.settings import DRIFT_THRESHOLDS
from drift_mlops.models.multi_model import MultiModelTrainer, ALGORITHMS, XGBOOST_AVAILABLE

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "saved_models")
HISTORY_DIR  = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.json")
DATA_DIR     = os.path.join(BASE_DIR, "real_data")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Drift Monitor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0a0a0d; color: #e4e4e7; }
section[data-testid="stSidebar"] { background:#111114; border-right:1px solid #1f1f24; }
#MainMenu, footer, header { visibility: hidden; }

.metric-card {
    background:#111114; border:1px solid #1f1f24;
    border-radius:8px; padding:18px 22px;
    position:relative; overflow:hidden;
}
.metric-card::before { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.metric-card.none::before     { background:#22c55e; }
.metric-card.low::before      { background:#3b82f6; }
.metric-card.medium::before   { background:#f59e0b; }
.metric-card.high::before     { background:#ef4444; }
.metric-card.critical::before { background:#dc2626; }
.metric-card.neutral::before  { background:#27272a; }

.metric-label { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#52525b; margin-bottom:6px; }
.metric-value { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; line-height:1; }
.metric-sub   { font-size:11px; color:#52525b; margin-top:4px; }

.step-label { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#52525b; background:#111114; border:1px solid #27272a; border-radius:3px; padding:1px 7px; display:inline-block; margin-bottom:6px; }
.step-title { font-size:15px; font-weight:600; color:#e4e4e7; margin-bottom:4px; }
.step-desc  { font-size:12px; color:#52525b; margin-bottom:14px; line-height:1.5; }

.alert-box  { border-radius:6px; padding:14px 18px; margin:10px 0; }
.alert-crit { background:rgba(220,38,38,.07); border:1px solid rgba(220,38,38,.2); border-left:3px solid #dc2626; }
.alert-warn { background:rgba(245,158,11,.07); border:1px solid rgba(245,158,11,.2); border-left:3px solid #f59e0b; }
.alert-ok   { background:rgba(34,197,94,.07);  border:1px solid rgba(34,197,94,.2);  border-left:3px solid #22c55e; }
.alert-info { background:rgba(59,130,246,.07); border:1px solid rgba(59,130,246,.2); border-left:3px solid #3b82f6; }
.alert-title{ font-weight:600; font-size:13px; margin-bottom:4px; }
.alert-body { font-size:12px; color:#a1a1aa; line-height:1.6; }

.divider { border:none; border-top:1px solid #1f1f24; margin:18px 0; }
.page-title { font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:600; letter-spacing:-.5px; }
.page-sub   { font-size:13px; color:#52525b; margin-top:3px; }

.live-badge {
    display:inline-block; background:rgba(239,68,68,.15);
    border:1px solid rgba(239,68,68,.3); color:#ef4444;
    font-family:'IBM Plex Mono',monospace; font-size:10px;
    padding:2px 10px; border-radius:20px; margin-left:10px;
    letter-spacing:1px; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

.event-line { font-family:'IBM Plex Mono',monospace; font-size:11px; padding:3px 0; color:#71717a; border-bottom:1px solid #1f1f24; }
.event-line.drift  { color:#ef4444; }
.event-line.retrain{ color:#22c55e; }
.event-line.ok     { color:#3b82f6; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

SEVERITY_COLORS = {
    "none":"#22c55e","low":"#3b82f6",
    "medium":"#f59e0b","high":"#ef4444","critical":"#dc2626"
}
PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#71717a", size=10),
    margin=dict(l=0, r=0, t=20, b=0),
)

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

for k, v in {
    # Analysis tab
    "ref_df": None, "new_df": None, "new_data_name": "",
    "models": None, "champion_algo": None,
    "active_algo": None, "trainer": None,
    "model_source": None, "feature_cols": None,
    "target_col": None, "analysis_result": None,
    # Demo tab
    "demo_running": False,
    "demo_models": None, "demo_trainer": None,
    "demo_champion": None,
    "demo_batches": [],      # [{batch_id, drift_score, severity, f1_rf, f1_xgb, f1_lr, event}]
    "demo_events": [],       # log
    "demo_ref_df": None,
    "demo_stream_df": None,
    "demo_stream_idx": 0,
    "demo_retrains": 0,
    # Common
    "slack_url": "",
    "notify_levels": ["high","critical"],
    "retrain_count": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

def save_models(models, trainer, champion, label="") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vdir = os.path.join(MODELS_DIR, ts)
    os.makedirs(vdir, exist_ok=True)
    saved = []
    for algo, r in models.items():
        if "model" not in r: continue
        with open(os.path.join(vdir, f"{algo}.pkl"), "wb") as f:
            pickle.dump(r["model"], f)
        saved.append(algo)
    if trainer and trainer.scalers:
        with open(os.path.join(vdir, "scalers.pkl"), "wb") as f:
            pickle.dump(trainer.scalers, f)
    meta = {
        "version": ts, "champion": champion, "algorithms": saved,
        "label": label, "saved_at": datetime.now().isoformat(),
        "metrics": {a: r.get("metrics",{}) for a,r in models.items() if "metrics" in r}
    }
    with open(os.path.join(vdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return ts

def list_saved_models():
    versions = []
    if not os.path.exists(MODELS_DIR): return versions
    for d in sorted(os.listdir(MODELS_DIR), reverse=True):
        meta_path = os.path.join(MODELS_DIR, d, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                versions.append(json.load(f))
    return versions

def load_model_version(version):
    vdir = os.path.join(MODELS_DIR, version)
    with open(os.path.join(vdir, "meta.json")) as f:
        meta = json.load(f)
    models = {}
    for algo in meta["algorithms"]:
        path = os.path.join(vdir, f"{algo}.pkl")
        if os.path.exists(path):
            with open(path,"rb") as f:
                model = pickle.load(f)
            models[algo] = {"model":model,"metrics":meta["metrics"].get(algo,{}),"algorithm":algo,"label":ALGORITHMS.get(algo,{}).get("label",algo)}
    scalers = {}
    sp = os.path.join(vdir,"scalers.pkl")
    if os.path.exists(sp):
        with open(sp,"rb") as f:
            scalers = pickle.load(f)
    return models, scalers, meta

def save_to_history(result, new_data_name=""):
    history = load_history()
    report  = result["drift_report"]
    active  = st.session_state.active_algo
    ref_m   = result.get("ref_metrics_all",{}).get(active,{})
    new_m   = result.get("new_metrics_all",{}).get(active,{})
    entry = {
        "timestamp": result["timestamp"],
        "new_data":  new_data_name,
        "drift_score": round(report.composite_score,4),
        "severity":    report.severity.value,
        "drifted_features": report.drifted_features,
        "n_drifted":  len(report.drifted_features),
        "n_total":    len(st.session_state.feature_cols or []),
        "active_algorithm": active,
        "champion":   st.session_state.champion_algo,
        "ref_f1": ref_m.get("f1"),
        "new_f1": new_m.get("f1"),
        "all_algorithms": {
            algo: {"ref_f1": result.get("ref_metrics_all",{}).get(algo,{}).get("f1"),
                   "new_f1": result.get("new_metrics_all",{}).get(algo,{}).get("f1")}
            for algo in (result.get("ref_metrics_all") or {})
        },
    }
    history.append(entry)
    with open(HISTORY_FILE,"w") as f:
        json.dump(history, f, indent=2, default=str)

def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE) as f: return json.load(f)
    except: return []

# ═══════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════════

def clean_df(df, target_col):
    df_c = df.select_dtypes(include=[np.number]).copy()
    if target_col not in df_c.columns:
        df_c[target_col] = df[target_col]
    df_c = df_c.dropna()
    vals = df_c[target_col]
    df_c[target_col] = (vals > vals.min()).astype(int) if len(vals.unique()) <= 2 else (vals > vals.median()).astype(int)
    return df_c

def get_drift(ref_df, new_df, feature_cols):
    detector = StatisticalDriftDetector()
    th = {"ks_p_value":DRIFT_THRESHOLDS.ks_p_value,
          "psi_threshold":DRIFT_THRESHOLDS.psi_medium,
          "wasserstein_threshold":DRIFT_THRESHOLDS.wasserstein_threshold}
    stat_results = detector.detect_drift(ref_df[feature_cols], new_df[feature_cols], thresholds=th)
    report = DriftScorer().score(stat_results)
    feat_scores = {f: sum(1 for t in tests if t.is_drifted)/len(tests)
                   for f,tests in stat_results.items()}
    return report, stat_results, feat_scores

def eval_all(models, trainer, X, y):
    if not models or not trainer: return {}
    return trainer.evaluate_on_new_data(models, X, y)

def run_full_analysis(ref_df, new_df, feature_cols, target_col):
    Xr,Xn = ref_df[feature_cols], new_df[feature_cols]
    yr,yn = ref_df[target_col],   new_df[target_col]
    report, stat_results, feat_scores = get_drift(ref_df, new_df, feature_cols)
    ref_eval = eval_all(st.session_state.models, st.session_state.trainer, Xr, yr)
    new_eval = eval_all(st.session_state.models, st.session_state.trainer, Xn, yn)
    return {"drift_report":report,"stat_results":stat_results,
            "ref_metrics_all":ref_eval,"new_metrics_all":new_eval,
            "feature_scores":feat_scores,"timestamp":datetime.now().isoformat()}

def do_retrain(new_weight):
    rdf,ndf = st.session_state.ref_df, st.session_state.new_df
    fc,tc   = st.session_state.feature_cols, st.session_state.target_col
    n_ref   = int(len(rdf)*(1-new_weight))
    combined = pd.concat([rdf.sample(n=min(n_ref,len(rdf)),random_state=42), ndf], ignore_index=True)
    t = MultiModelTrainer()
    results = t.train_all(combined[fc], combined[tc])
    champ   = t.get_champion(results, metric="f1")
    st.session_state.retrain_count += 1
    version = save_models(results, t, champ, f"Retrain #{st.session_state.retrain_count}")
    st.session_state.models  = results
    st.session_state.trainer = t
    st.session_state.champion_algo = champ
    st.session_state.active_algo   = champ
    return {"version":version,"results":results,"champion":champ,
            "n_rows":len(combined),"n_ref":n_ref,"n_new":len(ndf)}

def send_slack(url, result):
    report = result["drift_report"]
    sev    = report.severity.value
    icons  = {"none":"✅","low":"🔵","medium":"🟡","high":"🔴","critical":"🚨"}
    active = st.session_state.active_algo or "—"
    ref_f1 = result["ref_metrics_all"].get(active,{}).get("f1","?")
    new_f1 = result["new_metrics_all"].get(active,{}).get("f1","?")
    msg = {"blocks":[
        {"type":"header","text":{"type":"plain_text","text":f"{icons.get(sev,'⚪')} Drift Monitor Alert"}},
        {"type":"section","fields":[
            {"type":"mrkdwn","text":f"*Severity*\n{sev.upper()}"},
            {"type":"mrkdwn","text":f"*Score*\n{report.composite_score:.3f}"},
            {"type":"mrkdwn","text":f"*Model*\n{ALGORITHMS.get(active,{}).get('label',active)}"},
            {"type":"mrkdwn","text":f"*F1*\n{ref_f1} → {new_f1}"},
        ]},
        {"type":"section","text":{"type":"mrkdwn","text":f"*Affected*\n{', '.join(report.drifted_features[:5]) or 'None'}"}},
        {"type":"context","elements":[{"type":"mrkdwn","text":f"_{result['timestamp'][:19]}_"}]},
    ]}
    try:
        req = urllib.request.Request(url,json.dumps(msg).encode(),{"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=10) as r: return r.status==200
    except: return False

def step_ui(num, title, desc):
    st.markdown(f'<div class="step-label">STEP {num}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-desc">{desc}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DEMO HELPERS
# ═══════════════════════════════════════════════════════════════════

DEMO_REF_FILE   = os.path.join(DATA_DIR, "airline_reference_sample.csv")
DEMO_EARLY_FILE = os.path.join(DATA_DIR, "airline_new_early_sample.csv")
DEMO_LATE_FILE  = os.path.join(DATA_DIR, "airline_new_late_sample.csv")
DEMO_BATCH_SIZE = 3000
DEMO_TARGET     = "target"

def demo_log(msg, kind="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.demo_events.insert(0, {"ts":ts,"msg":msg,"kind":kind})
    if len(st.session_state.demo_events) > 50:
        st.session_state.demo_events = st.session_state.demo_events[:50]

def demo_process_batch():
    """Stream'den bir batch al, drift hesapla, gerekirse retrain yap."""
    stream = st.session_state.demo_stream_df
    idx    = st.session_state.demo_stream_idx
    ref_df = st.session_state.demo_ref_df
    models = st.session_state.demo_models
    trainer= st.session_state.demo_trainer
    fc     = [c for c in ref_df.columns if c != DEMO_TARGET]

    if idx >= len(stream):
        return False  # bitti

    batch = stream.iloc[idx:idx+DEMO_BATCH_SIZE].reset_index(drop=True)
    st.session_state.demo_stream_idx += DEMO_BATCH_SIZE

    # Drift hesapla
    report, _, _ = get_drift(ref_df, batch, fc)
    sev   = report.severity.value
    score = round(report.composite_score, 3)
    bid   = len(st.session_state.demo_batches)

    # F1 her model için
    Xb = batch[fc]
    yb = batch[DEMO_TARGET]
    f1s = {}
    for algo, r in models.items():
        if "model" not in r: continue
        try:
            if algo == "logistic_regression" and trainer and algo in trainer.scalers:
                Xs = trainer.scalers[algo].transform(Xb)
                yp = r["model"].predict(Xs)
            else:
                yp = r["model"].predict(Xb)
            f1s[algo] = round(f1_score(yb, yp, zero_division=0), 4)
        except:
            f1s[algo] = None

    # Event
    event = None
    if sev in ["high","critical"]:
        demo_log(f"Batch {bid}: drift={score} [{sev.upper()}] — AUTO RETRAIN triggered", "retrain")
        event = "RETRAIN"
        # Retrain
        n_ref = int(len(ref_df)*0.3)
        combined = pd.concat([ref_df.sample(n=n_ref,random_state=bid), batch], ignore_index=True)
        t2 = MultiModelTrainer()
        results2 = t2.train_all(combined[fc], combined[DEMO_TARGET])
        champ2   = t2.get_champion(results2,"f1")
        save_models(results2, t2, champ2, f"Demo retrain batch {bid}")
        st.session_state.demo_models   = results2
        st.session_state.demo_trainer  = t2
        st.session_state.demo_champion = champ2
        st.session_state.demo_retrains += 1
        models = results2
        trainer= t2
        # Recalculate f1 after retrain
        for algo, r in models.items():
            if "model" not in r: continue
            try:
                if algo == "logistic_regression" and t2 and algo in t2.scalers:
                    Xs = t2.scalers[algo].transform(Xb)
                    yp = r["model"].predict(Xs)
                else:
                    yp = r["model"].predict(Xb)
                f1s[algo] = round(f1_score(yb, yp, zero_division=0), 4)
            except:
                pass
    elif sev == "medium":
        demo_log(f"Batch {bid}: drift={score} [MEDIUM] — monitoring", "drift")
        event = "ALERT"
    else:
        demo_log(f"Batch {bid}: drift={score} [{sev.upper()}]", "ok")

    st.session_state.demo_batches.append({
        "batch_id": bid,
        "drift_score": score,
        "severity": sev,
        "event": event,
        **{f"f1_{algo}": f1s.get(algo) for algo in ["random_forest","xgboost","logistic_regression"]},
    })
    return True


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

col_title, col_slack = st.columns([4,1])
with col_title:
    st.markdown('<div class="page-title">◈ Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">MLOps · Real-Time Data Drift Detection & Mitigation</div>', unsafe_allow_html=True)
with col_slack:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("🔔 Slack"):
        slack_url = st.text_input("Webhook URL", value=st.session_state.slack_url,
            placeholder="https://hooks.slack.com/...", type="password", label_visibility="collapsed")
        if slack_url: st.session_state.slack_url = slack_url
        st.session_state.notify_levels = st.multiselect(
            "Alert on", ["low","medium","high","critical"],
            default=st.session_state.notify_levels)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_demo, tab_analysis, tab_history, tab_models = st.tabs([
    "🚀 Live Demo", "📊 Analysis", "📈 History", "💾 Saved Models"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: LIVE DEMO
# ══════════════════════════════════════════════════════════════════

with tab_demo:
    has_demo_data = all(os.path.exists(f) for f in [DEMO_REF_FILE, DEMO_EARLY_FILE, DEMO_LATE_FILE])

    if not has_demo_data:
        st.markdown("""
        <div class="alert-box alert-warn">
            <div class="alert-title" style="color:#f59e0b;">⚠ Demo data not found</div>
            <div class="alert-body">
                Run <code>python drift_mlops/prepare_airline.py</code> and
                <code>python drift_mlops/sample_airline.py</code> first.<br>
                Expected files in <code>drift_mlops/real_data/</code>:<br>
                airline_reference_sample.csv · airline_new_early_sample.csv · airline_new_late_sample.csv
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2,1,1,1])

    with col_ctrl1:
        st.markdown("### Live Demo")
        st.caption("Watch drift detection in real-time. System trains on winter data, then summer data streams in — triggering drift and automatic retraining.")

    with col_ctrl2:
        if st.button("▶ Start Demo", type="primary", use_container_width=True,
                     disabled=st.session_state.demo_running):
            # Load data
            ref_df    = pd.read_csv(DEMO_REF_FILE)
            early_df  = pd.read_csv(DEMO_EARLY_FILE)
            late_df   = pd.read_csv(DEMO_LATE_FILE)
            fc = [c for c in ref_df.columns if c != DEMO_TARGET]

            # Train initial models
            trainer = MultiModelTrainer()
            results = trainer.train_all(ref_df[fc], ref_df[DEMO_TARGET])
            champ   = trainer.get_champion(results,"f1")
            version = save_models(results, trainer, champ, "Demo initial training")

            # Stream = early + late
            stream = pd.concat([early_df, late_df], ignore_index=True)

            st.session_state.demo_ref_df    = ref_df
            st.session_state.demo_stream_df = stream
            st.session_state.demo_stream_idx= 0
            st.session_state.demo_models    = results
            st.session_state.demo_trainer   = trainer
            st.session_state.demo_champion  = champ
            st.session_state.demo_batches   = []
            st.session_state.demo_events    = []
            st.session_state.demo_retrains  = 0
            st.session_state.demo_running   = True

            demo_log(f"System started. Model trained on {len(ref_df):,} winter records.", "ok")
            demo_log(f"Champion: {ALGORITHMS.get(champ,{}).get('label',champ)}", "ok")
            for algo, r in results.items():
                if "metrics" in r:
                    demo_log(f"  {ALGORITHMS.get(algo,{}).get('label',algo)}: F1={r['metrics'].get('f1','?')}", "ok")
            st.rerun()

    with col_ctrl3:
        if st.button("⏭ Next Batch", use_container_width=True,
                     disabled=not st.session_state.demo_running):
            more = demo_process_batch()
            if not more:
                st.session_state.demo_running = False
                demo_log("All data processed. Demo complete.", "ok")
            st.rerun()

    with col_ctrl4:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["demo_running","demo_models","demo_trainer","demo_champion",
                      "demo_batches","demo_events","demo_ref_df","demo_stream_df",
                      "demo_stream_idx","demo_retrains"]:
                st.session_state[k] = False if k=="demo_running" else ([] if "batch" in k or "event" in k else (0 if "idx" in k or "retrain" in k else None))
            st.rerun()

    # Auto-run toggle
    auto_run = st.toggle("⚡ Auto-run (streams batches automatically)", value=False)

    if st.session_state.demo_running and auto_run and st.session_state.demo_batches is not None:
        more = demo_process_batch()
        if not more:
            st.session_state.demo_running = False
            demo_log("All data processed. Demo complete.", "ok")
        time.sleep(0.5)
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if not st.session_state.demo_batches:
        st.markdown("""
        <div style="background:#111114;border:1px solid #1f1f24;border-radius:8px;padding:40px;text-align:center;">
            <div style="font-size:32px;margin-bottom:12px;">✈</div>
            <div style="font-size:16px;font-weight:600;margin-bottom:8px;">Airline Delay Drift Demo</div>
            <div style="font-size:13px;color:#52525b;max-width:500px;margin:0 auto;line-height:1.6;">
                Model is trained on <strong>winter flight data</strong> (Jan–Apr 2018).<br>
                Summer data (May–Aug) then streams in batch by batch.<br>
                Watch drift scores rise, F1 drop, and automatic retraining kick in.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        batches = st.session_state.demo_batches

        # KPI row
        latest   = batches[-1]
        n_batches= len(batches)
        n_retrain= st.session_state.demo_retrains
        processed= st.session_state.demo_stream_idx
        total    = len(st.session_state.demo_stream_df) if st.session_state.demo_stream_df is not None else 1

        c1,c2,c3,c4,c5 = st.columns(5)
        sev = latest["severity"]
        sc  = latest["drift_score"]
        color = SEVERITY_COLORS.get(sev,"#52525b")

        with c1:
            st.markdown(f"""
            <div class="metric-card {sev}">
                <div class="metric-label">Drift Score</div>
                <div class="metric-value" style="color:{color};">{sc}</div>
                <div class="metric-sub">latest batch</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card {sev}">
                <div class="metric-label">Severity</div>
                <div class="metric-value" style="color:{color};font-size:20px;">{sev.upper()}</div>
                <div class="metric-sub">&nbsp;</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            champ = st.session_state.demo_champion
            f1_val = latest.get(f"f1_{champ}")
            f1_str = f"{f1_val:.4f}" if f1_val else "—"
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">Champion F1</div>
                <div class="metric-value" style="color:#e4e4e7;">{f1_str}</div>
                <div class="metric-sub">{ALGORITHMS.get(champ,{}).get('label',champ) if champ else '—'}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">Retrains</div>
                <div class="metric-value" style="color:#22c55e;">{n_retrain}</div>
                <div class="metric-sub">auto-triggered</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            pct = int(processed/total*100) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card neutral">
                <div class="metric-label">Progress</div>
                <div class="metric-value" style="color:#e4e4e7;">{pct}%</div>
                <div class="metric-sub">{processed:,} / {total:,} rows</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(processed/total, 1.0))

        # Charts row
        cl, cr = st.columns([3,2], gap="large")

        with cl:
            st.markdown("**Drift Score Trend**")
            df_b = pd.DataFrame(batches)

            fig = go.Figure()
            # Drift score line
            fig.add_trace(go.Scatter(
                x=df_b["batch_id"], y=df_b["drift_score"],
                mode="lines+markers", name="Drift Score",
                line=dict(color="#e4e4e7", width=2),
                marker=dict(
                    color=[SEVERITY_COLORS.get(s,"#71717a") for s in df_b["severity"]],
                    size=8, line=dict(width=0)
                ),
            ))
            # Threshold lines
            fig.add_hline(y=0.30, line_dash="dot", line_color="#3b82f6", line_width=1,
                          annotation_text="low", annotation_font=dict(size=9,color="#3b82f6"))
            fig.add_hline(y=0.50, line_dash="dot", line_color="#f59e0b", line_width=1,
                          annotation_text="medium", annotation_font=dict(size=9,color="#f59e0b"))
            fig.add_hline(y=0.70, line_dash="dot", line_color="#ef4444", line_width=1,
                          annotation_text="high", annotation_font=dict(size=9,color="#ef4444"))
            # Retrain markers
            retrain_batches = df_b[df_b["event"]=="RETRAIN"]
            if not retrain_batches.empty:
                fig.add_trace(go.Scatter(
                    x=retrain_batches["batch_id"], y=retrain_batches["drift_score"],
                    mode="markers", name="Retrain",
                    marker=dict(symbol="star", size=14, color="#22c55e"),
                ))

            fig.update_layout(**PLOT, height=280,
                xaxis=dict(title="Batch", gridcolor="#1a1a1f", dtick=1),
                yaxis=dict(range=[0,1], gridcolor="#1a1a1f"),
                legend=dict(orientation="h",yanchor="bottom",y=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
                showlegend=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            # F1 per algorithm
            st.markdown("**F1 Score per Algorithm**")
            fig2 = go.Figure()
            algo_colors = {"random_forest":"#3b82f6","xgboost":"#f59e0b","logistic_regression":"#22c55e"}
            for algo in ["random_forest","xgboost","logistic_regression"]:
                col_key = f"f1_{algo}"
                if col_key in df_b.columns:
                    vals = df_b[col_key].dropna()
                    if not vals.empty:
                        fig2.add_trace(go.Scatter(
                            x=df_b.loc[vals.index,"batch_id"],
                            y=vals,
                            mode="lines+markers",
                            name=ALGORITHMS.get(algo,{}).get("label",algo),
                            line=dict(color=algo_colors[algo], width=2),
                            marker=dict(size=5),
                        ))
            fig2.update_layout(**PLOT, height=220,
                xaxis=dict(title="Batch",gridcolor="#1a1a1f",dtick=1),
                yaxis=dict(range=[0,1],gridcolor="#1a1a1f"),
                legend=dict(orientation="h",yanchor="bottom",y=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        with cr:
            st.markdown("**Event Log**")
            events = st.session_state.demo_events
            for e in events[:20]:
                kind_class = {"retrain":"retrain","drift":"drift","ok":"ok"}.get(e["kind"],"")
                st.markdown(
                    f'<div class="event-line {kind_class}">{e["ts"]} · {e["msg"]}</div>',
                    unsafe_allow_html=True
                )

            # Algorithm comparison table
            if len(batches) > 0:
                st.markdown("<br>**Algorithm Comparison (Latest Batch)**")
                latest_b = batches[-1]
                rows = []
                champ = st.session_state.demo_champion
                for algo in ["random_forest","xgboost","logistic_regression"]:
                    f1 = latest_b.get(f"f1_{algo}")
                    if f1 is not None:
                        rows.append({
                            "Algorithm": ALGORITHMS.get(algo,{}).get("label",algo) + (" 🏆" if algo==champ else ""),
                            "F1": f1,
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: ANALYSIS
# ══════════════════════════════════════════════════════════════════

with tab_analysis:

    # STEP 1
    step_ui("01","Reference Data","Upload the dataset your model was originally trained on.")
    ref_file = st.file_uploader("", type=["csv"], key="ref_up", label_visibility="collapsed")
    if ref_file:
        try:
            ref_raw = pd.read_csv(ref_file)
            c1,c2 = st.columns([3,1])
            with c1:
                target_col = st.selectbox("Target column", ref_raw.columns.tolist(), key="tgt")
            with c2:
                st.markdown("<br>",unsafe_allow_html=True)
                st.caption(f"{len(ref_raw):,} rows")
            ref_df = clean_df(ref_raw, target_col)
            feature_cols = [c for c in ref_df.columns if c != target_col]
            st.session_state.ref_df = ref_df
            st.session_state.target_col = target_col
            st.session_state.feature_cols = feature_cols
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows",f"{len(ref_df):,}")
            c2.metric("Features",len(feature_cols))
            c3.metric("Class balance",f"0:{(ref_df[target_col]==0).sum()} / 1:{(ref_df[target_col]==1).sum()}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('<hr class="divider">',unsafe_allow_html=True)

    # STEP 2
    step_ui("02","Models","Train 3 algorithms in parallel, load a saved version, or upload your own .pkl.")
    model_choice = st.radio("",["multi","load","upload"],
        format_func=lambda x:{"multi":"Auto-train (RF + XGBoost + LogReg)","load":"Load saved version","upload":"Upload .pkl"}[x],
        horizontal=True,label_visibility="collapsed")

    if model_choice=="upload":
        mf = st.file_uploader("Model (.pkl)",type=["pkl"],key="model_up")
        if mf:
            try:
                model = pickle.load(mf)
                t = MultiModelTrainer()
                st.session_state.models = {"uploaded":{"model":model,"metrics":{},"algorithm":"uploaded","label":type(model).__name__}}
                st.session_state.trainer = t
                st.session_state.active_algo = st.session_state.champion_algo = "uploaded"
                st.session_state.model_source = "uploaded"
                st.success(f"Loaded: {type(model).__name__}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif model_choice=="load":
        versions = list_saved_models()
        if not versions:
            st.info("No saved models yet.")
        else:
            opts = {f"{v.get('label') or v['version']} — champion: {ALGORITHMS.get(v['champion'],{}).get('label',v['champion'])} ({v['saved_at'][:16]})":v['version'] for v in versions}
            sel_label = st.selectbox("Select version",list(opts.keys()))
            if st.button("Load",type="primary"):
                models,scalers,meta = load_model_version(opts[sel_label])
                t = MultiModelTrainer(); t.scalers = scalers
                st.session_state.models = models; st.session_state.trainer = t
                st.session_state.champion_algo = st.session_state.active_algo = meta["champion"]
                st.session_state.model_source = "loaded"
                st.success(f"Loaded {opts[sel_label]}"); st.rerun()

    elif model_choice=="multi":
        if st.session_state.ref_df is not None:
            if st.button("Train All Models",type="primary"):
                with st.spinner("Training 3 algorithms..."):
                    t = MultiModelTrainer()
                    X = st.session_state.ref_df[st.session_state.feature_cols]
                    y = st.session_state.ref_df[st.session_state.target_col]
                    results = t.train_all(X,y)
                    champ   = t.get_champion(results,"f1")
                    version = save_models(results,t,champ,"Initial training")
                    st.session_state.models = results; st.session_state.trainer = t
                    st.session_state.champion_algo = st.session_state.active_algo = champ
                    st.session_state.model_source = "auto_trained"
                st.success(f"Trained & saved → `{version}`")
                rows = []
                for algo,r in results.items():
                    if "metrics" not in r: continue
                    m = r["metrics"]
                    rows.append({"Algorithm":ALGORITHMS[algo]["label"]+(" 🏆" if algo==champ else ""),
                                 "Accuracy":m.get("accuracy","—"),"F1":m.get("f1","—"),"AUC-ROC":m.get("auc_roc","—")})
                if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        else:
            st.caption("Upload reference data first.")

    if st.session_state.models and st.session_state.model_source in ["auto_trained","loaded"]:
        avail = [a for a in st.session_state.models if "metrics" in st.session_state.models[a]]
        if len(avail) > 1:
            active = st.selectbox("Active model:",avail,
                index=avail.index(st.session_state.active_algo) if st.session_state.active_algo in avail else 0,
                format_func=lambda x:ALGORITHMS.get(x,{}).get("label",x)+(" 🏆" if x==st.session_state.champion_algo else ""))
            st.session_state.active_algo = active

    if st.session_state.models:
        al = ALGORITHMS.get(st.session_state.active_algo,{}).get("label",st.session_state.active_algo)
        st.caption(f"✓ Active: **{al}** ({st.session_state.model_source})")

    st.markdown('<hr class="divider">',unsafe_allow_html=True)

    # STEP 3
    step_ui("03","New Data","Upload recent data to check for drift.")
    new_file = st.file_uploader("",type=["csv"],key="new_up",label_visibility="collapsed")
    new_data_name = ""
    if new_file and st.session_state.target_col:
        new_data_name = new_file.name
        try:
            new_raw = pd.read_csv(new_file)
            new_df  = clean_df(new_raw, st.session_state.target_col)
            miss    = [c for c in st.session_state.feature_cols if c not in new_df.columns]
            if miss: st.error(f"Missing: {miss}")
            else:
                st.session_state.new_df = new_df[st.session_state.feature_cols+[st.session_state.target_col]]
                st.caption(f"✓ {len(new_df):,} rows loaded")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('<hr class="divider">',unsafe_allow_html=True)

    ready = all([st.session_state.ref_df is not None,
                 st.session_state.new_df  is not None,
                 st.session_state.models  is not None])
    if not ready:
        miss = [l for l,r in [("reference data",st.session_state.ref_df is not None),
                               ("models",st.session_state.models is not None),
                               ("new data",st.session_state.new_df is not None)] if not r]
        st.caption(f"Waiting for: {', '.join(miss)}")

    if st.button("Run Analysis →",type="primary",disabled=not ready):
        with st.spinner("Analyzing..."):
            result = run_full_analysis(st.session_state.ref_df,st.session_state.new_df,
                                       st.session_state.feature_cols,st.session_state.target_col)
            st.session_state.analysis_result = result
            save_to_history(result, new_data_name)
        if st.session_state.slack_url:
            sev = result["drift_report"].severity.value
            if sev in st.session_state.notify_levels:
                ok = send_slack(st.session_state.slack_url, result)
                st.toast("Slack sent ✓" if ok else "Slack failed ✗")
        st.rerun()

    # Results
    if not st.session_state.analysis_result:
        st.markdown('<div style="background:#111114;border:1px solid #1f1f24;border-radius:8px;padding:40px;text-align:center;"><div style="font-size:28px;color:#27272a;margin-bottom:10px;">◈</div><div style="font-size:14px;font-weight:600;margin-bottom:6px;">No analysis yet</div><div style="font-size:12px;color:#52525b;">Complete steps 1–3 and click Run Analysis.</div></div>', unsafe_allow_html=True)
        st.stop()

    result   = st.session_state.analysis_result
    report   = result["drift_report"]
    severity = report.severity.value
    color    = SEVERITY_COLORS.get(severity,"#52525b")
    active   = st.session_state.active_algo
    ref_m    = result["ref_metrics_all"].get(active,{})
    new_m    = result["new_metrics_all"].get(active,{})
    ref_f1   = ref_m.get("f1"); new_f1 = new_m.get("f1")
    f1_delta = round(new_f1-ref_f1,4) if (ref_f1 and new_f1) else None
    n_drifted= len(report.drifted_features)
    n_total  = len(st.session_state.feature_cols)

    st.markdown('<hr class="divider">',unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#52525b;margin-bottom:12px;">Results · {result["timestamp"][:19].replace("T"," ")}</div>',unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card {severity}"><div class="metric-label">Drift Score</div><div class="metric-value" style="color:{color};">{report.composite_score:.3f}</div><div class="metric-sub">out of 1.000</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card {severity}"><div class="metric-label">Severity</div><div class="metric-value" style="color:{color};font-size:20px;">{severity.upper()}</div><div class="metric-sub">&nbsp;</div></div>',unsafe_allow_html=True)
    with c3:
        dsub = (f'<div style="color:{"#22c55e" if f1_delta>=0 else "#ef4444"};font-size:11px;">{"↑" if f1_delta>=0 else "↓"} {abs(f1_delta):.4f} vs ref</div>' if f1_delta is not None else '<div class="metric-sub">—</div>')
        f1c  = "#22c55e" if (f1_delta and f1_delta>=0) else "#ef4444"
        al   = ALGORITHMS.get(active,{}).get("label",active)
        st.markdown(f'<div class="metric-card neutral"><div class="metric-label">{al} F1</div><div class="metric-value" style="color:{f1c};">{f"{new_f1:.4f}" if new_f1 else "—"}</div>{dsub}</div>',unsafe_allow_html=True)
    with c4:
        fc2 = "#ef4444" if n_drifted>n_total//2 else "#f59e0b" if n_drifted>0 else "#22c55e"
        st.markdown(f'<div class="metric-card neutral"><div class="metric-label">Drifted Features</div><div class="metric-value" style="color:{fc2};">{n_drifted}<span style="font-size:16px;color:#52525b;">/{n_total}</span></div><div class="metric-sub">&nbsp;</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # Algorithm comparison
    if st.session_state.model_source in ["auto_trained","loaded"] and len(result["new_metrics_all"])>1:
        st.markdown("**Algorithm Comparison Under Drift**")
        rows=[]
        for algo in result["new_metrics_all"]:
            rm=result["ref_metrics_all"].get(algo,{}); nm=result["new_metrics_all"].get(algo,{})
            if "f1" not in rm or "f1" not in nm: continue
            d=round(nm["f1"]-rm["f1"],4)
            rows.append({"Algorithm":ALGORITHMS.get(algo,{}).get("label",algo)+(" 🏆" if algo==st.session_state.champion_algo else ""),
                         "F1 (Reference)":rm["f1"],"F1 (New Data)":nm["f1"],
                         "Δ F1":f"{d:+.4f}","AUC-ROC":nm.get("auc_roc","—")})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # Alert
    if severity in ["high","critical"]:
        feats=", ".join(report.drifted_features[:6]) or "—"
        st.markdown(f'<div class="alert-box alert-crit"><div class="alert-title" style="color:#ef4444;">⚠ High drift</div><div class="alert-body">Performance degraded. <strong>Retrain recommended.</strong><br>Affected: {feats}</div></div>',unsafe_allow_html=True)
    elif severity=="medium":
        feats=", ".join(report.drifted_features[:4]) or "—"
        st.markdown(f'<div class="alert-box alert-warn"><div class="alert-title" style="color:#f59e0b;">◎ Moderate drift</div><div class="alert-body">Monitor closely. Retrain if F1 continues to drop.<br>Affected: {feats}</div></div>',unsafe_allow_html=True)
    elif severity=="low":
        st.markdown('<div class="alert-box alert-info"><div class="alert-title" style="color:#3b82f6;">○ Low drift</div><div class="alert-body">Minor shift detected. Model still performing adequately. You can retrain below if desired.</div></div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-ok"><div class="alert-title" style="color:#22c55e;">✓ No significant drift</div><div class="alert-body">Models performing within expected bounds.</div></div>',unsafe_allow_html=True)

    st.markdown('<hr class="divider">',unsafe_allow_html=True)

    # Charts
    cl,cr = st.columns(2,gap="large")
    with cl:
        st.markdown("**Feature Drift**")
        fs=result["feature_scores"]; fl=sorted(fs,key=lambda x:fs[x]); sc=[fs[f] for f in fl]
        cols_=[("#ef4444" if s>0.5 else "#f59e0b" if s>0.3 else "#3b82f6") for s in sc]
        fig=go.Figure(go.Bar(x=sc,y=fl,orientation="h",marker=dict(color=cols_,opacity=0.85)))
        fig.add_vline(x=0.5,line_dash="dot",line_color="#52525b",line_width=1,
                      annotation_text="threshold",annotation_font=dict(size=9,color="#52525b"))
        fig.update_layout(**PLOT,height=max(200,len(fl)*30),
                          xaxis=dict(range=[0,1],gridcolor="#1a1a1f"),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"),bargap=0.4)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with cr:
        st.markdown("**Distribution Comparison**")
        all_f=list(fs.keys()); top=max(fs,key=fs.get) if fs else all_f[0]
        sel=st.selectbox("",all_f,index=all_f.index(top),label_visibility="collapsed",key="dist_sel")
        fig2=go.Figure()
        fig2.add_trace(go.Histogram(x=st.session_state.ref_df[sel],name="Reference",opacity=0.7,marker_color="#3b82f6",nbinsx=40,histnorm="probability density"))
        fig2.add_trace(go.Histogram(x=st.session_state.new_df[sel],name="New Data",opacity=0.7,marker_color="#ef4444",nbinsx=40,histnorm="probability density"))
        fig2.update_layout(**PLOT,height=220,barmode="overlay",
                           xaxis=dict(gridcolor="#1a1a1f"),yaxis=dict(gridcolor="#1a1a1f"),
                           legend=dict(orientation="h",yanchor="bottom",y=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

    # Model performance
    st.markdown('<hr class="divider">',unsafe_allow_html=True)
    st.markdown("**Model Performance**")
    pcols=st.columns(3)
    for i,(key,lbl) in enumerate([("accuracy","Accuracy"),("f1","F1 Score"),("auc_roc","AUC-ROC")]):
        if key in ref_m and key in new_m:
            d=round(new_m[key]-ref_m[key],4)
            pcols[i].metric(lbl,f"{new_m[key]:.4f}",delta=f"{d:+.4f}",delta_color="normal")

    with st.expander("Statistical Test Details"):
        rows=[]
        for feat,tests in result["stat_results"].items():
            for t in tests:
                rows.append({"Feature":feat,"Test":t.test_name,"Statistic":round(t.statistic,5),
                             "P-Value":round(t.p_value,5) if t.p_value else "—",
                             "Result":"Drift" if t.is_drifted else "OK"})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # MITIGATION — always visible
    st.markdown('<hr class="divider">',unsafe_allow_html=True)
    st.markdown("**Mitigation — Retrain & Save Models**")
    if severity in ["high","critical"]:
        st.caption("⚠ High drift detected — retraining strongly recommended.")
    else:
        st.caption("You can retrain to adapt models to the new data distribution at any time.")

    cs,ci=st.columns([2,1])
    with cs:
        nw=st.slider("New data weight",0.1,0.9,0.7,0.1,key="mit_slider",
                     help="Higher = model learns more from new data.")
    with ci:
        st.markdown("<br>",unsafe_allow_html=True)
        st.caption(f"Reference {int((1-nw)*100)}% · New {int(nw*100)}%")

    if st.button("Retrain & Save All Models",type="primary",key="retrain_btn"):
        with st.spinner("Retraining 3 algorithms and saving..."):
            ret=do_retrain(nw)
        st.success(f"✓ Retrained & saved — version `{ret['version']}`")
        st.caption(f"Trained on {ret['n_rows']:,} rows · {ret['n_ref']:,} ref + {ret['n_new']:,} new · Held-out 20% test")
        rows=[]
        for algo,r in ret["results"].items():
            if "metrics" not in r: continue
            old_f1=result["new_metrics_all"].get(algo,{}).get("f1",0)
            nf=r["metrics"].get("f1",0)
            imp=round(nf-old_f1,4)
            rows.append({"Algorithm":ALGORITHMS.get(algo,{}).get("label",algo)+(" 🏆" if algo==ret["champion"] else ""),
                         "F1 (before)":old_f1,"F1 (after)":nf,
                         "Improvement":f"{imp:+.4f}","AUC-ROC":r["metrics"].get("auc_roc","—")})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        champ_m=ret["results"][ret["champion"]]["model"]
        st.download_button(
            f"↓ Download Champion ({ALGORITHMS.get(ret['champion'],{}).get('label',ret['champion'])}) .pkl",
            data=pickle.dumps(champ_m),
            file_name=f"model_{ret['champion']}_{ret['version']}.pkl",
            mime="application/octet-stream",
        )

    st.markdown('<hr class="divider">',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    def make_csv():
        rows=[{"feature":f,"drift_ratio":round(s,4),"drifted":s>0.5} for f,s in result["feature_scores"].items()]
        return pd.DataFrame(rows).sort_values("drift_ratio",ascending=False).to_csv(index=False)
    with c1:
        st.download_button("↓ Export CSV",data=make_csv(),
            file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",use_container_width=True)
    with c2:
        sm={"timestamp":result["timestamp"],"drift_score":report.composite_score,"severity":severity,
            "drifted_features":report.drifted_features,"champion":st.session_state.champion_algo,
            "ref_metrics":result["ref_metrics_all"],"new_metrics":result["new_metrics_all"]}
        st.download_button("↓ Export JSON",data=json.dumps(sm,indent=2,default=str),
            file_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",use_container_width=True)
    with c3:
        if st.session_state.slack_url:
            if st.button("↗ Send to Slack",use_container_width=True):
                ok=send_slack(st.session_state.slack_url,result)
                (st.success if ok else st.error)("Sent!" if ok else "Failed.")


# ══════════════════════════════════════════════════════════════════
# TAB 3: HISTORY
# ══════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("**Analysis History**")
    st.caption("All runs saved to disk. Persists across sessions.")
    history=load_history()
    if not history:
        st.info("No history yet. Run an analysis to start tracking.")
    else:
        df_h=pd.DataFrame(history)
        df_h["timestamp"]=pd.to_datetime(df_h["timestamp"])
        df_h=df_h.sort_values("timestamp")

        st.markdown("**Drift Score Over Time**")
        fig_t=go.Figure()
        for sev,col in SEVERITY_COLORS.items():
            mask=df_h["severity"]==sev
            if mask.any():
                sub=df_h[mask]
                fig_t.add_trace(go.Scatter(x=sub["timestamp"],y=sub["drift_score"],
                    mode="markers",name=sev.upper(),marker=dict(color=col,size=8)))
        for y,col,lbl in [(0.30,"#3b82f6","low"),(0.50,"#f59e0b","medium"),(0.70,"#ef4444","high")]:
            fig_t.add_hline(y=y,line_dash="dot",line_color=col,line_width=1,
                            annotation_text=lbl,annotation_font=dict(size=9,color=col))
        fig_t.update_layout(**PLOT,height=260,
            xaxis=dict(gridcolor="#1a1a1f"),yaxis=dict(gridcolor="#1a1a1f",range=[0,1]),
            legend=dict(orientation="h",yanchor="bottom",y=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_t,use_container_width=True,config={"displayModeBar":False})

        st.markdown("**All Runs**")
        rows=[]
        for h in reversed(history):
            active=h.get("active_algorithm","—")
            rf=h.get("ref_f1"); nf=h.get("new_f1")
            d=round(nf-rf,4) if isinstance(rf,float) and isinstance(nf,float) else "—"
            rows.append({"Time":h["timestamp"][:19].replace("T"," "),"Data":h.get("new_data","—"),
                         "Drift":h["drift_score"],"Severity":h["severity"].upper(),
                         "Drifted":f"{h['n_drifted']}/{h.get('n_total','?')}",
                         "Model":ALGORITHMS.get(active,{}).get("label",active),
                         "F1 ref":rf,"F1 new":nf,"Δ F1":d})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        if "all_algorithms" in (history[0] if history else {}):
            st.markdown("**F1 per Algorithm Over Time**")
            fig_f=go.Figure()
            ac={"random_forest":"#3b82f6","xgboost":"#f59e0b","logistic_regression":"#22c55e"}
            for algo in ["random_forest","xgboost","logistic_regression"]:
                xs,ys=[],[]
                for h in history:
                    v=h.get("all_algorithms",{}).get(algo,{}).get("new_f1")
                    if v is not None:
                        xs.append(pd.to_datetime(h["timestamp"])); ys.append(v)
                if xs:
                    fig_f.add_trace(go.Scatter(x=xs,y=ys,mode="lines+markers",
                        name=ALGORITHMS.get(algo,{}).get("label",algo),
                        line=dict(color=ac.get(algo,"#71717a"),width=2),marker=dict(size=5)))
            fig_f.update_layout(**PLOT,height=230,
                xaxis=dict(gridcolor="#1a1a1f"),yaxis=dict(gridcolor="#1a1a1f",range=[0,1]),
                legend=dict(orientation="h",yanchor="bottom",y=1,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_f,use_container_width=True,config={"displayModeBar":False})

        if st.button("Clear History"):
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.success("Cleared."); st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 4: SAVED MODELS
# ══════════════════════════════════════════════════════════════════

with tab_models:
    st.markdown("**Saved Model Versions**")
    st.caption("All trained and retrained models are saved automatically.")
    versions=list_saved_models()
    if not versions:
        st.info("No saved models yet. Train models in the Analysis tab.")
    else:
        for v in versions:
            with st.expander(f"**{v.get('label') or v['version']}** — saved {v['saved_at'][:16]}"):
                c1,c2=st.columns(2)
                with c1:
                    st.markdown(f"**Version:** `{v['version']}`")
                    st.markdown(f"**Champion:** {ALGORITHMS.get(v['champion'],{}).get('label',v['champion'])}")
                    st.markdown(f"**Algorithms:** {', '.join(v['algorithms'])}")
                with c2:
                    if v.get("metrics"):
                        rows=[{"Algorithm":ALGORITHMS.get(a,{}).get("label",a)+(" 🏆" if a==v["champion"] else ""),
                               "F1":m.get("f1","—"),"AUC-ROC":m.get("auc_roc","—")}
                              for a,m in v["metrics"].items()]
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                vdir=os.path.join(MODELS_DIR,v["version"])
                dcols=st.columns(len(v["algorithms"]))
                for i,algo in enumerate(v["algorithms"]):
                    path=os.path.join(vdir,f"{algo}.pkl")
                    if os.path.exists(path):
                        with open(path,"rb") as f: data=f.read()
                        dcols[i].download_button(
                            f"↓ {ALGORITHMS.get(algo,{}).get('label',algo)}",
                            data=data,file_name=f"{algo}_{v['version']}.pkl",
                            mime="application/octet-stream",use_container_width=True,
                            key=f"dl_{v['version']}_{algo}")