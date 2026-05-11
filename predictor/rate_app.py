"""
Rate Forecaster App
===================
Haftalik/donemsel gecikme orani tahmini.

Klasik binary classification'dan farkli:
  - Her satir icin 0/1 degil, donem icin oran tahmini
  - Regression modelleri kullanir
  - Leakage yok: sadece ucustan ONCE bilinenler kullanilir
  - Metrik: MAE (yuzde puan), R2

Calistir:
  streamlit run predictor/rate_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from predictor.engine import (
    RateForecaster, FORECAST_ALGORITHMS,
    ForecastTrainResult, ForecastPrediction,
    DriftDetector, ModelStore, HistoryStore,
)
from predictor.engine.forecaster import XGBOOST_AVAILABLE

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR    = os.path.join(BASE_DIR, "saved_rate")
HISTORY_FILE = os.path.join(BASE_DIR, "history_rate.json")
DATA_DIR     = os.path.join(os.path.dirname(BASE_DIR), "drift_mlops", "real_data")

os.makedirs(SAVED_DIR, exist_ok=True)

forecaster = RateForecaster()
detector   = DriftDetector()

# ── Demo dosyalari ─────────────────────────────────────────────────
DEMO_REF   = os.path.join(DATA_DIR, "airline_rate_reference.csv")
DEMO_EARLY = os.path.join(DATA_DIR, "airline_rate_early.csv")
DEMO_LATE  = os.path.join(DATA_DIR, "airline_rate_late.csv")
DEMO_FULL  = os.path.join(DATA_DIR, "airline_rate_full.csv")

# Cok yillik veri
DEMO_TRAIN = os.path.join(DATA_DIR, "airline_rate_train.csv")
DEMO_VAL   = os.path.join(DATA_DIR, "airline_rate_val.csv")
DEMO_TEST  = os.path.join(DATA_DIR, "airline_rate_test.csv")
DEMO_ALL   = os.path.join(DATA_DIR, "airline_rate_all.csv")

HAS_MULTIYEAR = all(os.path.exists(f) for f in [DEMO_TRAIN, DEMO_VAL, DEMO_TEST])

TARGET_COL = "delay_rate"

# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rate Forecaster", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
*, html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; }
.stApp { background:#08080b; color:#e4e4e7; }
#MainMenu, footer, header { visibility:hidden; }

.card { background:#0f0f12; border:1px solid #1a1a20; border-radius:10px; padding:18px 20px; position:relative; overflow:hidden; margin-bottom:4px; }
.card-accent::before { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.card-blue::before   { background:#3b82f6; }
.card-purple::before { background:#a855f7; }
.card-green::before  { background:#22c55e; }
.card-yellow::before { background:#f59e0b; }
.card-red::before    { background:#ef4444; }
.card-neutral::before{ background:#27272a; }
.card-label { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:2.5px; text-transform:uppercase; color:#52525b; margin-bottom:8px; }
.card-value { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; line-height:1; margin-bottom:4px; }
.card-sub   { font-size:11px; color:#52525b; }

.section-title { font-size:12px; font-weight:600; color:#71717a; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #1a1a20; }
.divider { border:none; border-top:1px solid #1a1a20; margin:18px 0; }
.step-num   { font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:2px; color:#52525b; background:#0f0f12; border:1px solid #1a1a20; border-radius:3px; padding:1px 8px; display:inline-block; margin-bottom:6px; }
.step-title { font-size:14px; font-weight:600; margin-bottom:3px; }
.step-desc  { font-size:12px; color:#52525b; margin-bottom:12px; line-height:1.5; }
.algo-row   { display:flex; align-items:center; gap:10px; padding:9px 12px; background:#0f0f12; border:1px solid #1a1a20; border-radius:7px; margin-bottom:6px; }
.algo-name  { font-weight:600; font-size:12px; flex:1; }
.algo-val   { font-family:'IBM Plex Mono',monospace; font-size:12px; color:#52525b; }
.insight-box { background:#0f0f12; border:1px solid #1a1a20; border-left:3px solid #a855f7; border-radius:8px; padding:14px 18px; margin:10px 0; }
.insight-title { font-weight:600; font-size:12px; margin-bottom:4px; color:#a855f7; }
.insight-body  { font-size:12px; color:#a1a1aa; line-height:1.7; }
.info-box { background:rgba(59,130,246,.06); border:1px solid rgba(59,130,246,.2); border-radius:8px; padding:12px 16px; margin:8px 0; font-size:12px; color:#a1a1aa; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#71717a", size=10),
    margin=dict(l=0, r=0, t=24, b=0),
)

for k, v in {
    "ref_df":None,"ref_name":"",
    "forecast_results":None,"champion":None,"model_version":None,
    "new_df":None,"new_name":"","has_target":False,
    "pred_results":None,"drift_result":None,"analysis":None,
}.items():
    if k not in st.session_state: st.session_state[k] = v

def step_ui(num, title, desc):
    st.markdown(f'<div class="step-num">STEP {num}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-desc">{desc}</div>', unsafe_allow_html=True)

def section(title): st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
def divider():      st.markdown('<hr class="divider">', unsafe_allow_html=True)

def save_forecast_model(results, champion, label=""):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    vdir = os.path.join(SAVED_DIR, ts)
    os.makedirs(vdir, exist_ok=True)
    for algo, r in results.items():
        with open(os.path.join(vdir, f"{algo}.pkl"), "wb") as f:
            pickle.dump(r, f)
    import json
    meta = {
        "version":    ts, "champion": champion,
        "algorithms": list(results.keys()), "label": label,
        "saved_at":   datetime.now().isoformat(),
        "metrics": {a: {"mae":r.mae,"rmse":r.rmse,"r2":r.r2,"cv_mae":r.cv_mae}
                    for a,r in results.items()},
    }
    with open(os.path.join(vdir,"meta.json"),"w") as f:
        json.dump(meta, f, indent=2)
    return ts

def generate_insight(champ_pred, ref_mean, drift_result):
    pr = champ_pred.mean_predicted
    ar = champ_pred.mean_actual
    lines = []

    if ar is not None:
        err = abs(ar - pr)
        if err <= 1.5:
            lines.append(f"Model predicted **{pr:.1f}%** delay rate, actual was **{ar:.1f}%** — excellent accuracy (error: {err:.1f}pp).")
        elif err <= 3:
            lines.append(f"Model predicted **{pr:.1f}%** delay rate, actual was **{ar:.1f}%** — good accuracy (error: {err:.1f}pp).")
        else:
            lines.append(f"Model predicted **{pr:.1f}%** delay rate, actual was **{ar:.1f}%** — error is {err:.1f}pp. Drift may be contributing.")
    else:
        lines.append(f"Model predicts **{pr:.1f}%** delay rate for this period.")

    if drift_result:
        sev = drift_result.severity
        if sev in ["high","critical"]:
            lines.append(f"**{sev.upper()} drift detected** — the new period's flight patterns differ significantly from training data. Retraining recommended.")
        elif sev == "medium":
            lines.append(f"Moderate drift detected — monitor and consider retraining.")
        else:
            lines.append(f"No significant drift — model is operating in a familiar distribution.")

    if ar is not None:
        shift = ar - ref_mean
        if abs(shift) > 3:
            direction = "higher" if shift > 0 else "lower"
            lines.append(f"Delay rate is **{abs(shift):.1f}pp {direction}** than the training period ({ref_mean:.1f}%).")

    return " ".join(lines)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:21px;font-weight:600;letter-spacing:-.5px;padding:4px 0 2px;">📈 Rate Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:12px;color:#52525b;margin-bottom:4px;">Weekly flight delay rate forecasting — no data leakage — real ML</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>How this works:</strong> Instead of predicting per-flight (0/1), this model learns from weekly aggregated patterns
— flight volume, average distance, time-of-day distribution, carrier diversity — all known <em>before</em> flights depart.
It then forecasts the delay rate for a new period. This is genuine rate forecasting with no data leakage.
</div>
""", unsafe_allow_html=True)

divider()

tab_main, tab_full, tab_models = st.tabs(["📊 Forecast", "📅 Full Year View", "💾 Models"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: FORECAST
# ══════════════════════════════════════════════════════════════════

with tab_main:
    col_left, col_right = st.columns([1,2], gap="large")

    with col_left:
        # STEP 1
        step_ui("01","Reference Period",
                "Load weekly aggregated data from the reference period (Jan-Apr 2018).")

        has_demo = os.path.exists(DEMO_REF)
        if HAS_MULTIYEAR:
            st.markdown("""
            <div class="info-box" style="border-left:3px solid #22c55e;">
            <strong>Multi-year data available!</strong> 413 weeks of training data (2009-2016).
            Use these for a real ML model.
            </div>""", unsafe_allow_html=True)
            if st.button("⚡ Load 2009-2016 Training (413 weeks)", use_container_width=True, key="multi_ref"):
                df = pd.read_csv(DEMO_TRAIN)
                st.session_state.ref_df   = df
                st.session_state.ref_name = "airline_rate_train.csv (2009-2016)"
                st.rerun()

        if has_demo:
            if st.button("⚡ Load Demo Reference (Jan-Apr 2018 only)", use_container_width=True):
                df = pd.read_csv(DEMO_REF)
                st.session_state.ref_df   = df
                st.session_state.ref_name = "airline_rate_reference.csv"
                st.rerun()

        ref_file = st.file_uploader("Or upload weekly CSV", type=["csv"],
                                    key="ref_up", label_visibility="collapsed")
        if ref_file:
            st.session_state.ref_df   = pd.read_csv(ref_file)
            st.session_state.ref_name = ref_file.name
            st.rerun()

        if st.session_state.ref_df is not None:
            rdf = st.session_state.ref_df
            if TARGET_COL in rdf.columns:
                ref_mean = round(rdf[TARGET_COL].mean()*100, 2)
                ref_std  = round(rdf[TARGET_COL].std()*100, 2)
                c1,c2,c3 = st.columns(3)
                c1.metric("Weeks", len(rdf))
                c2.metric("Avg delay rate", f"{ref_mean:.1f}%")
                c3.metric("Std dev", f"{ref_std:.1f}pp")

                # Mini trend chart
                fig_ref = go.Figure(go.Scatter(
                    x=rdf["week_number"] if "week_number" in rdf.columns else list(range(len(rdf))),
                    y=rdf[TARGET_COL]*100,
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=5),
                    fill="tozeroy", fillcolor="rgba(59,130,246,.08)",
                ))
                fig_ref.update_layout(**PLOT, height=120,
                    xaxis=dict(gridcolor="#1a1a20", title="Week"),
                    yaxis=dict(gridcolor="#1a1a20", title="Delay %"),
                )
                st.plotly_chart(fig_ref, use_container_width=True,
                                config={"displayModeBar":False})

        divider()

        # STEP 2
        step_ui("02","Train Forecast Models",
                "3 regression models learn the relationship between flight patterns and delay rates.")

        if st.session_state.ref_df is not None and TARGET_COL in st.session_state.ref_df.columns:
            if st.button("🚀 Train All Models", type="primary",
                         use_container_width=True, key="train_btn"):
                with st.spinner("Training RF · XGBoost · Linear Regression..."):
                    rdf  = st.session_state.ref_df
                    fc   = [c for c in rdf.columns if c != TARGET_COL]
                    X    = rdf[fc]
                    y    = rdf[TARGET_COL]
                    results = forecaster.train_all(X, y)
                    champ   = forecaster.get_champion(results)
                    version = save_forecast_model(results, champ,
                                f"Training on {st.session_state.ref_name}")
                    st.session_state.forecast_results = results
                    st.session_state.champion = champ
                    st.session_state.model_version = version
                st.success(f"Saved → `{version}`")
                st.rerun()
        else:
            st.caption("Load reference data first.")

        if st.session_state.forecast_results:
            champ = st.session_state.champion
            for algo, r in st.session_state.forecast_results.items():
                is_champ = algo == champ
                fc2 = "#f59e0b" if is_champ else "#52525b"
                st.markdown(f"""
                <div class="algo-row">
                    <span>{"⭐" if is_champ else "○"}</span>
                    <div class="algo-name">{FORECAST_ALGORITHMS[algo]['label']}</div>
                    <div class="algo-val" style="color:{fc2};">CV MAE {r.cv_mae:.2f}pp</div>
                    <div class="algo-val">R² {r.r2:.3f}</div>
                </div>""", unsafe_allow_html=True)

            # Feature importance (champion)
            champ_result = st.session_state.forecast_results[champ]
            imp_df = forecaster.get_feature_importance(champ_result)
            if imp_df is not None:
                with st.expander("📊 Feature Importance"):
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df["importance"],
                        y=imp_df["feature"],
                        orientation="h",
                        marker_color="#3b82f6",
                        opacity=0.85,
                    ))
                    fig_imp.update_layout(**PLOT, height=max(150,len(imp_df)*25),
                        xaxis=dict(gridcolor="#1a1a20"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                        bargap=0.4,
                    )
                    st.plotly_chart(fig_imp, use_container_width=True,
                                    config={"displayModeBar":False})

        divider()

        # STEP 3
        step_ui("03","New Period Data",
                "Load weekly data for the period you want to forecast. Target optional.")

        if HAS_MULTIYEAR:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⚡ 2017 Validation", use_container_width=True, key="demo_val"):
                    df = pd.read_csv(DEMO_VAL)
                    st.session_state.new_df     = df
                    st.session_state.new_name   = "2017 (validation)"
                    st.session_state.has_target = TARGET_COL in df.columns
                    st.session_state.pred_results = None
                    st.rerun()
            with c2:
                if st.button("⚡ 2018 Test", use_container_width=True, key="demo_test"):
                    df = pd.read_csv(DEMO_TEST)
                    st.session_state.new_df     = df
                    st.session_state.new_name   = "2018 (test)"
                    st.session_state.has_target = TARGET_COL in df.columns
                    st.session_state.pred_results = None
                    st.rerun()

        if has_demo:
            c1,c2 = st.columns(2)
            with c1:
                if st.button("⚡ May-Aug 2018", use_container_width=True, key="demo_early"):
                    df = pd.read_csv(DEMO_EARLY)
                    st.session_state.new_df   = df
                    st.session_state.new_name = "May-Aug 2018"
                    st.session_state.has_target = TARGET_COL in df.columns
                    st.session_state.pred_results = None
                    st.rerun()
            with c2:
                if st.button("⚡ Sep-Dec 2018", use_container_width=True, key="demo_late"):
                    df = pd.read_csv(DEMO_LATE)
                    st.session_state.new_df   = df
                    st.session_state.new_name = "Sep-Dec 2018"
                    st.session_state.has_target = TARGET_COL in df.columns
                    st.session_state.pred_results = None
                    st.rerun()

        new_file = st.file_uploader("Or upload weekly CSV", type=["csv"],
                                    key="new_up", label_visibility="collapsed")
        if new_file:
            df = pd.read_csv(new_file)
            st.session_state.new_df     = df
            st.session_state.new_name   = new_file.name
            st.session_state.has_target = TARGET_COL in df.columns
            st.rerun()

        if st.session_state.new_df is not None:
            ndf = st.session_state.new_df
            c1,c2 = st.columns(2)
            c1.metric("Weeks", len(ndf))
            if TARGET_COL in ndf.columns:
                c2.metric("Actual avg rate", f"{ndf[TARGET_COL].mean()*100:.1f}%")
            else:
                c2.metric("Target", "Not available")

        divider()

        ready = all([
            st.session_state.ref_df is not None,
            st.session_state.forecast_results is not None,
            st.session_state.new_df is not None,
        ])
        if not ready:
            miss = [l for l,ok in [
                ("reference data", st.session_state.ref_df is not None),
                ("trained models", st.session_state.forecast_results is not None),
                ("new data",       st.session_state.new_df is not None),
            ] if not ok]
            st.caption(f"Missing: {', '.join(miss)}")

        if st.button("▶ Run Forecast & Analysis", type="primary",
                     use_container_width=True, disabled=not ready):
            with st.spinner("Forecasting..."):
                rdf     = st.session_state.ref_df
                ndf     = st.session_state.new_df
                results = st.session_state.forecast_results
                champ   = st.session_state.champion
                fc      = [c for c in rdf.columns if c != TARGET_COL]

                avail  = [c for c in fc if c in ndf.columns]
                X_new  = ndf[avail]
                y_new  = ndf[TARGET_COL] if TARGET_COL in ndf.columns else None

                pred_results = forecaster.predict_all(results, X_new, y_new)
                champ_pred   = pred_results.get(champ)

                # Drift
                ref_fc = [c for c in fc if c != TARGET_COL]
                dr = detector.detect(
                    ref_df=rdf, new_df=ndf,
                    feature_cols=[c for c in ref_fc if c in ndf.columns],
                    ref_rate=rdf[TARGET_COL].mean()*100 if TARGET_COL in rdf.columns else None,
                    new_rate=champ_pred.mean_actual if champ_pred else None,
                )

                st.session_state.pred_results  = pred_results
                st.session_state.drift_result  = dr
                st.session_state.analysis = {
                    "ref_mean":   round(rdf[TARGET_COL].mean()*100, 2) if TARGET_COL in rdf.columns else 0,
                    "champ_pred": champ_pred,
                }
            st.rerun()

    # ── RIGHT PANEL ───────────────────────────────────────────────
    with col_right:
        if not st.session_state.pred_results:
            st.markdown("""
            <div style="background:#0f0f12;border:1px solid #1a1a20;border-radius:12px;
                        padding:60px;text-align:center;margin-top:40px;">
                <div style="font-size:40px;margin-bottom:14px;opacity:.2;">📈</div>
                <div style="font-size:16px;font-weight:600;margin-bottom:8px;">No forecast yet</div>
                <div style="font-size:13px;color:#52525b;line-height:1.8;max-width:420px;margin:0 auto;">
                    Complete the 3 steps, then click<br>
                    <strong>▶ Run Forecast & Analysis</strong>
                </div>
            </div>""", unsafe_allow_html=True)
            st.stop()

        pred_results = st.session_state.pred_results
        dr           = st.session_state.drift_result
        ana          = st.session_state.analysis
        champ        = st.session_state.champion
        champ_pred   = ana["champ_pred"]
        ref_mean     = ana["ref_mean"]

        SEVERITY_COLORS = {"none":"#22c55e","low":"#3b82f6","medium":"#f59e0b",
                           "high":"#ef4444","critical":"#dc2626"}
        sev_color = SEVERITY_COLORS.get(dr.severity,"#52525b")

        # ── FORECAST RESULTS ──────────────────────────────────────
        section("Forecast Results")

        if champ_pred.mean_actual is not None:
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="card card-accent card-blue"><div class="card-label">Reference Avg</div><div class="card-value" style="color:#3b82f6;">{ref_mean:.1f}<span style="font-size:16px;">%</span></div><div class="card-sub">Jan-Apr 2018</div></div>', unsafe_allow_html=True)
            with c2:
                pr = champ_pred.mean_predicted
                st.markdown(f'<div class="card card-accent card-purple"><div class="card-label">Predicted Rate</div><div class="card-value" style="color:#a855f7;">{pr:.1f}<span style="font-size:16px;">%</span></div><div class="card-sub">{FORECAST_ALGORITHMS[champ]["label"]} ⭐</div></div>', unsafe_allow_html=True)
            with c3:
                ar = champ_pred.mean_actual
                col = "#22c55e" if ar<20 else "#f59e0b" if ar<25 else "#ef4444"
                ec  = "green" if ar<20 else "yellow" if ar<25 else "red"
                st.markdown(f'<div class="card card-accent card-{ec}"><div class="card-label">Actual Rate</div><div class="card-value" style="color:{col};">{ar:.1f}<span style="font-size:16px;">%</span></div><div class="card-sub">Ground truth</div></div>', unsafe_allow_html=True)
            with c4:
                mae = champ_pred.mae
                mc  = "green" if mae<=1.5 else "yellow" if mae<=3 else "red"
                st.markdown(f'<div class="card card-accent card-{mc}"><div class="card-label">MAE</div><div class="card-value" style="color:{"#22c55e" if mae<=1.5 else "#f59e0b" if mae<=3 else "#ef4444"};">{mae:.2f}<span style="font-size:16px;">pp</span></div><div class="card-sub">Mean absolute error</div></div>', unsafe_allow_html=True)
        else:
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="card card-accent card-blue"><div class="card-label">Reference Avg</div><div class="card-value" style="color:#3b82f6;">{ref_mean:.1f}<span style="font-size:16px;">%</span></div><div class="card-sub">Training period</div></div>', unsafe_allow_html=True)
            with c2:
                pr = champ_pred.mean_predicted
                st.markdown(f'<div class="card card-accent card-purple"><div class="card-label">Predicted Rate</div><div class="card-value" style="color:#a855f7;">{pr:.1f}<span style="font-size:16px;">%</span></div><div class="card-sub">{FORECAST_ALGORITHMS[champ]["label"]} ⭐</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Week-by-week chart
        section("Week-by-Week Forecast")
        ndf = st.session_state.new_df
        weeks = ndf["week_number"].values if "week_number" in ndf.columns else list(range(len(ndf)))

        fig_wk = go.Figure()

        # Referans bölgesi
        rdf = st.session_state.ref_df
        if "week_number" in rdf.columns:
            ref_weeks = rdf["week_number"].values
            ref_rates = rdf[TARGET_COL].values * 100
            fig_wk.add_trace(go.Scatter(
                x=ref_weeks, y=ref_rates,
                mode="lines+markers", name="Reference",
                line=dict(color="#3b82f6", width=2, dash="dot"),
                marker=dict(size=4),
            ))

        # Tahmin
        for algo, pred in pred_results.items():
            is_champ = algo == champ
            color = {"rf_regressor":"#3b82f6","xgb_regressor":"#f59e0b","linear":"#22c55e"}.get(algo,"#71717a")
            fig_wk.add_trace(go.Scatter(
                x=weeks, y=pred.predicted_rates*100,
                mode="lines+markers" if is_champ else "lines",
                name=FORECAST_ALGORITHMS[algo]["label"] + (" ⭐" if is_champ else ""),
                line=dict(color=color, width=3 if is_champ else 1.5,
                          dash="solid" if is_champ else "dot"),
                marker=dict(size=6) if is_champ else {},
            ))

        # Gerçek
        if champ_pred.actual_rates is not None:
            fig_wk.add_trace(go.Scatter(
                x=weeks, y=champ_pred.actual_rates*100,
                mode="markers", name="Actual",
                marker=dict(color="#22c55e", size=10, symbol="diamond",
                            line=dict(color="white", width=1)),
            ))

        fig_wk.update_layout(**PLOT, height=300,
            xaxis=dict(title="Week of Year", gridcolor="#1a1a20"),
            yaxis=dict(title="Delay Rate (%)", gridcolor="#1a1a20"),
            legend=dict(orientation="h", yanchor="bottom", y=1,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_wk, use_container_width=True, config={"displayModeBar":False})

        divider()

        # Algorithm comparison table
        section("Algorithm Comparison")
        rows = []
        for algo, pred in pred_results.items():
            row = {
                "Algorithm":     FORECAST_ALGORITHMS[algo]["label"] + (" ⭐" if algo==champ else ""),
                "Predicted Avg": f"{pred.mean_predicted:.1f}%",
            }
            if pred.mean_actual is not None:
                row["Actual Avg"] = f"{pred.mean_actual:.1f}%"
                row["MAE (pp)"]   = f"{pred.mae:.2f}"
                row["RMSE (pp)"]  = f"{pred.rmse:.2f}"
                row["R²"]         = f"{pred.r2:.3f}"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # CV MAE from training
        st.caption("Cross-validation MAE from training:")
        cv_rows = []
        if st.session_state.forecast_results:
            for algo, r in st.session_state.forecast_results.items():
                cv_rows.append({
                    "Algorithm": FORECAST_ALGORITHMS[algo]["label"] + (" ⭐" if algo==champ else ""),
                    "CV MAE (pp)": f"{r.cv_mae:.2f}",
                    "R² (train)":  f"{r.r2:.3f}",
                    "Train mean":  f"{r.train_mean_rate:.1f}%",
                })
        if cv_rows:
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        divider()

        # Drift
        section("Drift Analysis")
        c1,c2,c3 = st.columns(3)
        with c1:
            sc = {"none":"green","low":"blue","medium":"yellow","high":"red","critical":"red"}.get(dr.severity,"neutral")
            st.markdown(f'<div class="card card-accent card-{sc}"><div class="card-label">Drift Score</div><div class="card-value" style="color:{sev_color};">{dr.drift_score:.3f}</div><div class="card-sub">out of 1.000</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card card-accent card-{sc}"><div class="card-label">Severity</div><div class="card-value" style="color:{sev_color};font-size:20px;">{dr.severity.upper()}</div><div class="card-sub">&nbsp;</div></div>', unsafe_allow_html=True)
        with c3:
            nd=dr.n_features_drifted; nt=dr.n_features_total
            dc="#ef4444" if nd>nt//2 else "#f59e0b" if nd>0 else "#22c55e"
            rd_str=f"{dr.rate_drift:.1f}pp" if dr.rate_drift else "—"
            st.markdown(f'<div class="card card-accent card-neutral"><div class="card-label">Drifted Features</div><div class="card-value" style="color:{dc};">{nd}<span style="font-size:16px;color:#52525b;">/{nt}</span></div><div class="card-sub">Rate shift: {rd_str}</div></div>', unsafe_allow_html=True)

        # Insight
        insight = generate_insight(champ_pred, ref_mean, dr)
        st.markdown(f'<div class="insight-box"><div class="insight-title">💡 Insight</div><div class="insight-body">{insight}</div></div>', unsafe_allow_html=True)

        # Retrain
        divider()
        section("Retrain")
        if dr.severity in ["high","critical"]:
            st.warning("High drift detected — retraining recommended.")

        ndf_ = st.session_state.new_df
        can_retrain = TARGET_COL in ndf_.columns if ndf_ is not None else False

        c1,c2 = st.columns([3,1])
        with c1: nw = st.slider("New data weight",0.1,0.9,0.7,0.1,key="nw_s")
        with c2: st.markdown("<br>",unsafe_allow_html=True); st.caption(f"Ref {int((1-nw)*100)}% · New {int(nw*100)}%")

        if not can_retrain: st.caption("ℹ Retrain requires new data with delay_rate column.")
        if st.button("🔄 Retrain & Save", type="primary", disabled=not can_retrain, key="retrain_btn"):
            with st.spinner("Retraining..."):
                rdf_ = st.session_state.ref_df
                fc_  = [c for c in rdf_.columns if c != TARGET_COL]
                avail_ = [c for c in fc_ if c in ndf_.columns]
                n_ref  = int(len(rdf_)*(1-nw))
                combined = pd.concat([
                    rdf_.sample(n=min(n_ref,len(rdf_)),random_state=42),
                    ndf_[avail_+[TARGET_COL]],
                ], ignore_index=True)
                X_,y_ = combined[avail_], combined[TARGET_COL]
                new_results = forecaster.train_all(X_,y_)
                new_champ   = forecaster.get_champion(new_results)
                version = save_forecast_model(new_results, new_champ,
                    f"Retrain on {st.session_state.new_name}")
                st.session_state.forecast_results = new_results
                st.session_state.champion = new_champ
                st.session_state.model_version = version
                # Eski sonuclari temizle — kullanici yeniden Run Forecast basacak
                st.session_state.pred_results = None
                st.session_state.analysis = None
                st.session_state.drift_result = None
            st.success(f"✓ Saved → `{version}`")
            st.info("Models updated. Load new data and click **▶ Run Forecast & Analysis** to see results with the new model.")
            rows=[]
            for algo,r in new_results.items():
                old = pred_results.get(algo)
                rows.append({
                    "Algorithm": FORECAST_ALGORITHMS[algo]["label"]+(" ⭐" if algo==new_champ else ""),
                    "Old CV MAE": f"{st.session_state.forecast_results[algo].cv_mae:.2f}pp" if algo in st.session_state.forecast_results else "—",
                    "New CV MAE": f"{r.cv_mae:.2f}pp",
                    "New R²": f"{r.r2:.3f}",
                })
            if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2: FULL YEAR VIEW
# ══════════════════════════════════════════════════════════════════

with tab_full:
    section("Full Year — 52 Weeks Overview")

    if not os.path.exists(DEMO_FULL):
        st.info("Run prepare_airline_rate.py first to generate airline_rate_full.csv")
        st.stop()

    full = pd.read_csv(DEMO_FULL)
    full["period"] = full["week_number"].apply(
        lambda w: "Jan-Apr (Ref)" if w<=17 else "May-Aug (Early)" if w<=34 else "Sep-Dec (Late)"
    )

    period_colors = {
        "Jan-Apr (Ref)":   "#3b82f6",
        "May-Aug (Early)": "#ef4444",
        "Sep-Dec (Late)":  "#22c55e",
    }

    fig_full = go.Figure()
    for period, color in period_colors.items():
        sub = full[full["period"]==period]
        fig_full.add_trace(go.Scatter(
            x=sub["week_number"], y=sub["delay_rate"]*100,
            mode="lines+markers", name=period,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.08)").replace("rgb","rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        ))

    fig_full.add_hline(y=full["delay_rate"].mean()*100, line_dash="dot",
                       line_color="#71717a",
                       annotation_text=f"Annual avg: {full['delay_rate'].mean()*100:.1f}%",
                       annotation_font=dict(size=10, color="#71717a"))

    fig_full.update_layout(**PLOT, height=350,
        xaxis=dict(title="Week of Year", gridcolor="#1a1a20", dtick=4),
        yaxis=dict(title="Delay Rate (%)", gridcolor="#1a1a20"),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="2018 Weekly Flight Delay Rate — All 52 Weeks",
                   font=dict(size=13, color="#a1a1aa")),
    )
    st.plotly_chart(fig_full, use_container_width=True, config={"displayModeBar":False})

    # Özet stats
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Jan-Apr avg", f"{full[full['week_number']<=17]['delay_rate'].mean()*100:.1f}%")
    c2.metric("May-Aug avg", f"{full[(full['week_number']>=18)&(full['week_number']<=34)]['delay_rate'].mean()*100:.1f}%")
    c3.metric("Sep-Dec avg", f"{full[full['week_number']>=35]['delay_rate'].mean()*100:.1f}%")
    c4.metric("Peak week", f"Week {full.loc[full['delay_rate'].idxmax(),'week_number']} ({full['delay_rate'].max()*100:.1f}%)")

    divider()
    section("Feature Patterns by Period")

    numeric_features = [c for c in full.columns if c not in [TARGET_COL,"period","week_number"]]
    sel_feat = st.selectbox("Select feature", numeric_features)

    fig_feat = go.Figure()
    for period, color in period_colors.items():
        sub = full[full["period"]==period]
        fig_feat.add_trace(go.Scatter(
            x=sub["week_number"], y=sub[sel_feat],
            mode="lines+markers", name=period,
            line=dict(color=color, width=2), marker=dict(size=5),
        ))
    fig_feat.update_layout(**PLOT, height=250,
        xaxis=dict(title="Week", gridcolor="#1a1a20"),
        yaxis=dict(title=sel_feat, gridcolor="#1a1a20"),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar":False})

    divider()
    st.markdown("**Raw Data**")
    st.dataframe(full, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3: MODELS
# ══════════════════════════════════════════════════════════════════

with tab_models:
    import json
    section("Saved Forecast Models")

    versions = []
    if os.path.exists(SAVED_DIR):
        for d in sorted(os.listdir(SAVED_DIR), reverse=True):
            mp = os.path.join(SAVED_DIR, d, "meta.json")
            if os.path.exists(mp):
                with open(mp) as f:
                    versions.append(json.load(f))

    if not versions:
        st.info("No saved models yet. Train models to start saving.")
    else:
        for v in versions:
            champ = v.get("champion","")
            with st.expander(f"**{v.get('label') or v['version']}** — {v['saved_at'][:16]}"):
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Version:** `{v['version']}`")
                    st.markdown(f"**Champion:** {FORECAST_ALGORITHMS.get(champ,{}).get('label',champ)}")
                with c2:
                    if v.get("metrics"):
                        rows=[{
                            "Algorithm": FORECAST_ALGORITHMS.get(a,{}).get("label",a)+(" ⭐" if a==champ else ""),
                            "CV MAE (pp)": m.get("cv_mae","—"),
                            "R²": m.get("r2","—"),
                        } for a,m in v["metrics"].items()]
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

                dcols = st.columns(len(v["algorithms"]))
                for i,algo in enumerate(v["algorithms"]):
                    path = os.path.join(SAVED_DIR, v["version"], f"{algo}.pkl")
                    if os.path.exists(path):
                        with open(path,"rb") as f: data=f.read()
                        dcols[i].download_button(
                            f"↓ {FORECAST_ALGORITHMS.get(algo,{}).get('label',algo)}",
                            data=data, file_name=f"{algo}_{v['version']}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=f"dl_{v['version']}_{algo}",
                        )