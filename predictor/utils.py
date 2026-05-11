"""
Utils
=====
Paylaşılan yardımcı fonksiyonlar.

- UI helpers (Streamlit markdown bileşenleri)
- Insight üretici (otomatik yorum)
- Grafik factory'leri
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Dict, List

from predictor.engine.types import DriftResult, PredictionResult
from predictor.config import ALGORITHMS, SEVERITY_COLORS


# ── Plot defaults ─────────────────────────────────────────────────

PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#71717a", size=10),
    margin=dict(l=0, r=0, t=24, b=0),
)


# ── UI Components ─────────────────────────────────────────────────

def step_ui(num: str, title: str, desc: str):
    """Step header."""
    st.markdown(f'<div class="step-num">STEP {num}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-desc">{desc}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str,
                sub: str = "", color: str = "#e4e4e7",
                accent: str = "neutral") -> str:
    return f"""
    <div class="card card-accent card-{accent}">
        <div class="card-label">{label}</div>
        <div class="card-value" style="color:{color};">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>"""


def rate_color(rate: float) -> str:
    """Oran değerine göre renk."""
    if rate < 15:   return "#22c55e"
    elif rate < 30: return "#f59e0b"
    else:           return "#ef4444"


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Insight Generator ─────────────────────────────────────────────

def generate_insight(drift: DriftResult,
                     champ_pred: PredictionResult,
                     event: str,
                     ref_rate: float,
                     unit: str = "%") -> str:
    """Analiz sonucuna göre otomatik yorum üret."""
    lines = []
    pr = champ_pred.predicted_rate
    ar = champ_pred.actual_rate
    is_reg = (unit != "%")

    def fmt(v):
        if v is None: return "—"
        if is_reg: return f"{v:,.0f} {unit}"
        return f"{v:.1f}%"

    def fmt_err(e):
        if e is None: return "—"
        if is_reg: return f"{e:,.0f} {unit}"
        return f"{e:.1f}pp"

    # 1. Tahmin doğruluğu
    if ar is not None:
        err = abs(ar - pr)
        err_pct = (err / abs(ar) * 100) if (is_reg and ar) else err
        t_good = 5 if is_reg else 2
        t_ok   = 15 if is_reg else 5
        if err_pct <= t_good:
            lines.append(f"Model predicted **{fmt(pr)}** {event}, actual was **{fmt(ar)}** — excellent accuracy (error: {fmt_err(err)}, {err_pct:.1f}%).")
        elif err_pct <= t_ok:
            lines.append(f"Model predicted **{fmt(pr)}** {event}, actual was **{fmt(ar)}** — acceptable accuracy (error: {fmt_err(err)}, {err_pct:.1f}%). Drift may be contributing.")
        else:
            lines.append(f"Model predicted **{fmt(pr)}** {event}, actual was **{fmt(ar)}** — significant error ({fmt_err(err)}, {err_pct:.1f}%). Data drift is likely causing model degradation.")
    else:
        lines.append(f"Model predicts **{fmt(pr)}** {event} for this dataset. Upload labeled data to validate accuracy.")

    # 2. Drift yorumu
    sev = drift.severity
    sc  = drift.drift_score
    if sev in ["high", "critical"]:
        top_feats = ", ".join(drift.drifted_features[:3]) or "—"
        lines.append(f"**{sev.upper()} drift detected** (score: {sc:.3f}). Distribution shifted significantly. Most affected: {top_feats}. **Retraining recommended.**")
    elif sev == "medium":
        lines.append(f"Moderate drift detected (score: {sc:.3f}). Monitor closely and consider retraining if accuracy declines.")
    else:
        lines.append(f"No significant drift detected (score: {sc:.3f}). Model operating within familiar distribution.")

    # 3. Value shift yorumu
    if drift.rate_drift is not None:
        rd  = drift.rate_drift
        cur = ar if ar is not None else pr
        threshold = 10 if is_reg else 2
        if rd > threshold:
            lines.append(f"The {event} shifted **{fmt_err(rd)}** from reference ({fmt(ref_rate)} → {fmt(cur)}). This is a meaningful change.")
        elif rd > threshold / 2:
            lines.append(f"The {event} shifted slightly ({fmt_err(rd)} from reference). Within acceptable range.")

    return " ".join(lines)


# ── Chart Factories ───────────────────────────────────────────────

def make_gauge(actual_rate: float, predicted_rate: float,
               ref_rate: float, event: str) -> go.Figure:
    """Gauge chart: gerçek oran, tahmin çizgisi, ref çizgisi."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=actual_rate,
        delta={"reference": ref_rate, "suffix": "pp vs ref",
               "valueformat": ".1f"},
        title={"text": f"Actual {event.title()} Rate",
               "font": {"size": 13, "color": "#a1a1aa"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#52525b"},
            "bar":  {"color": rate_color(actual_rate)},
            "steps": [
                {"range": [0,  15], "color": "rgba(34,197,94,.07)"},
                {"range": [15, 30], "color": "rgba(245,158,11,.07)"},
                {"range": [30, 100],"color": "rgba(239,68,68,.07)"},
            ],
            "threshold": {
                "line":      {"color": "#a855f7", "width": 2},
                "thickness": 0.75,
                "value":     predicted_rate,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(
        **{k: v for k, v in PLOT.items() if k != "margin"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=230,
    )
    return fig


def make_algo_bar(pred_results: Dict,
                  ref_rate: float,
                  event: str) -> go.Figure:
    """Her algoritmanın predicted vs actual rate bar chart."""
    algos  = list(pred_results.keys())
    labels = [ALGORITHMS[a]["label"] for a in algos]
    pred_r = [pred_results[a].predicted_rate for a in algos]
    act_r  = [pred_results[a].actual_rate for a in algos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pred_r, name="Predicted",
        marker_color="#a855f7", opacity=0.85,
    ))
    if any(v is not None for v in act_r):
        fig.add_trace(go.Bar(
            x=labels,
            y=[v if v is not None else 0 for v in act_r],
            name="Actual",
            marker_color="#22c55e", opacity=0.85,
        ))
    fig.add_hline(
        y=ref_rate,
        line_dash="dot", line_color="#3b82f6",
        annotation_text=f"Reference {ref_rate:.1f}%",
        annotation_font=dict(size=9, color="#3b82f6"),
    )
    fig.update_layout(
        **PLOT, height=220, barmode="group",
        xaxis=dict(gridcolor="#1a1a20"),
        yaxis=dict(gridcolor="#1a1a20",
                   title=f"{event.title()} Rate (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_feature_drift_bar(feature_scores: Dict[str, float]) -> go.Figure:
    """Feature drift horizontal bar chart."""
    fl = sorted(feature_scores, key=lambda x: feature_scores[x])
    sc = [feature_scores[f] for f in fl]
    colors = [
        "#ef4444" if s > 0.5 else
        "#f59e0b" if s > 0.3 else
        "#3b82f6"
        for s in sc
    ]
    fig = go.Figure(go.Bar(
        x=sc, y=fl, orientation="h",
        marker=dict(color=colors, opacity=0.85),
    ))
    fig.add_vline(
        x=0.5, line_dash="dot", line_color="#52525b", line_width=1,
        annotation_text="threshold",
        annotation_font=dict(size=9, color="#52525b"),
    )
    fig.update_layout(
        **PLOT,
        height=max(180, len(fl) * 28),
        xaxis=dict(range=[0, 1], gridcolor="#1a1a20"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        bargap=0.4,
        title=dict(text="Feature Drift Score",
                   font=dict(size=12, color="#a1a1aa")),
    )
    return fig


def make_distribution_chart(ref_series: pd.Series,
                             new_series: pd.Series,
                             feature_name: str) -> go.Figure:
    """İki dağılımı karşılaştıran histogram."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ref_series, name="Reference",
        opacity=0.7, marker_color="#3b82f6", nbinsx=40,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=new_series, name="New Data",
        opacity=0.7, marker_color="#ef4444", nbinsx=40,
        histnorm="probability density",
    ))
    fig.update_layout(
        **PLOT, height=220, barmode="overlay",
        xaxis=dict(title=feature_name, gridcolor="#1a1a20"),
        yaxis=dict(gridcolor="#1a1a20"),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Distribution Comparison",
                   font=dict(size=12, color="#a1a1aa")),
    )
    return fig


def make_rate_trend(history: List[Dict]) -> go.Figure:
    """Geçmişteki oran trendini göster."""
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()

    if "ref_rate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ref_rate"],
            mode="lines", name="Reference Rate",
            line=dict(color="#3b82f6", width=1, dash="dot"),
        ))
    if "predicted_rate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted_rate"],
            mode="lines+markers", name="Predicted Rate",
            line=dict(color="#a855f7", width=2),
            marker=dict(size=6),
        ))
    if "actual_rate" in df.columns:
        actual = df["actual_rate"].dropna()
        if not actual.empty:
            fig.add_trace(go.Scatter(
                x=df.loc[actual.index, "timestamp"],
                y=actual, mode="markers",
                name="Actual Rate",
                marker=dict(color="#22c55e", size=8, symbol="diamond"),
            ))

    fig.update_layout(
        **PLOT, height=260,
        xaxis=dict(gridcolor="#1a1a20"),
        yaxis=dict(gridcolor="#1a1a20", title="Rate (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Rate Trend Over Time",
                   font=dict(size=12, color="#a1a1aa")),
    )
    return fig


def make_drift_trend(history: List[Dict]) -> go.Figure:
    """Drift skoru trendini göster."""
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    sev_map = {
        "none":     "#22c55e",
        "low":      "#3b82f6",
        "medium":   "#f59e0b",
        "high":     "#ef4444",
        "critical": "#dc2626",
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["drift_score"],
        mode="lines+markers", name="Drift Score",
        line=dict(color="#e4e4e7", width=2),
        marker=dict(
            color=[sev_map.get(s, "#71717a") for s in df.get("severity", ["none"] * len(df))],
            size=8,
        ),
    ))

    for y, col, lbl in [
        (0.30, "#3b82f6", "low"),
        (0.50, "#f59e0b", "medium"),
        (0.70, "#ef4444", "high"),
    ]:
        fig.add_hline(
            y=y, line_dash="dot", line_color=col, line_width=1,
            annotation_text=lbl,
            annotation_font=dict(size=9, color=col),
        )

    fig.update_layout(
        **PLOT, height=220,
        xaxis=dict(gridcolor="#1a1a20"),
        yaxis=dict(gridcolor="#1a1a20", range=[0, 1]),
        title=dict(text="Drift Score Over Time",
                   font=dict(size=12, color="#a1a1aa")),
    )
    return fig