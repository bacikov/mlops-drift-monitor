"""
Drift Detection Dashboard (Streamlit)
======================================
Real-time pipeline görselleştirmesi ve interaktif kontrol paneli.

Çalıştırmak için:
    streamlit run drift_mlops/dashboard.py

Üç mod:
    1. Sentetik veri (kontrollü drift enjeksiyonu)
    2. CSV upload (kendi verini yükle)
    3. Benchmark veri setleri (UCI Electricity)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from drift_mlops.data.generator import DataGenerator, DriftConfig, DriftType
from drift_mlops.pipeline.orchestrator import DriftPipeline


# ═══════════════════════════════════════════════════════════════════
# SAYFA YAPILANDIRMASI
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Drift Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Özel CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .severity-none { color: #94a3b8; }
    .severity-low { color: #60a5fa; }
    .severity-medium { color: #fbbf24; }
    .severity-high { color: #f87171; }
    .severity-critical { color: #dc2626; font-weight: bold; }
    .event-log {
        font-family: 'Courier New', monospace;
        font-size: 11px;
        background: #0f172a;
        padding: 10px;
        border-radius: 6px;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "batch_history" not in st.session_state:
    st.session_state.batch_history = []
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "synthetic"
if "stream_data" not in st.session_state:
    st.session_state.stream_data = None
if "stream_idx" not in st.session_state:
    st.session_state.stream_idx = 0
if "reference_data" not in st.session_state:
    st.session_state.reference_data = None


# ═══════════════════════════════════════════════════════════════════
# HELPER FONKSIYONLAR
# ═══════════════════════════════════════════════════════════════════

def log_event(message: str, level: str = "info"):
    """Event log'a mesaj ekle."""
    icons = {"info": "ℹ️", "warn": "⚠️", "success": "✅", "error": "❌", "drift": "🎯"}
    icon = icons.get(level, "•")
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"{timestamp} {icon} {message}")
    # Sadece son 50 event'i tut
    st.session_state.event_log = st.session_state.event_log[:50]


def reset_pipeline():
    """Pipeline'ı sıfırla."""
    st.session_state.pipeline = None
    st.session_state.batch_history = []
    st.session_state.event_log = []
    st.session_state.stream_data = None
    st.session_state.stream_idx = 0
    st.session_state.reference_data = None


def prepare_synthetic_data(n_ref=3000, seed=42):
    """Sentetik veri modu için referans veri hazırla."""
    generator = DataGenerator(n_features=10, random_state=seed)
    X_ref, y_ref = generator.generate_reference(n_samples=n_ref)
    return generator, X_ref, y_ref


def prepare_csv_data(df: pd.DataFrame, target_col: str, ref_ratio: float = 0.6):
    """Yüklenen CSV'yi referans ve stream olarak böl."""
    split_idx = int(len(df) * ref_ratio)
    
    ref_df = df.iloc[:split_idx].reset_index(drop=True)
    stream_df = df.iloc[split_idx:].reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c != target_col]
    
    X_ref = ref_df[feature_cols]
    y_ref = ref_df[target_col]
    X_stream = stream_df[feature_cols]
    y_stream = stream_df[target_col]
    
    return X_ref, y_ref, X_stream, y_stream


def get_severity_color(severity: str) -> str:
    colors = {
        "none": "#94a3b8",
        "low": "#60a5fa",
        "medium": "#fbbf24",
        "high": "#f87171",
        "critical": "#dc2626",
    }
    return colors.get(severity, "#94a3b8")


def get_severity_emoji(severity: str) -> str:
    emojis = {
        "none": "🟢",
        "low": "🔵",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴",
    }
    return emojis.get(severity, "⚪")


# ═══════════════════════════════════════════════════════════════════
# BATCH PROCESSING FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════

def process_batches_synthetic(drift_type_str, magnitude, concept_drift, n_batches, batch_size):
    """Sentetik veri için batch üret ve işle."""
    pipeline = st.session_state.pipeline
    generator = st.session_state.generator
    
    drift_type_map = {
        "none": DriftType.NONE,
        "sudden": DriftType.SUDDEN,
        "gradual": DriftType.GRADUAL,
        "incremental": DriftType.INCREMENTAL,
        "recurring": DriftType.RECURRING,
    }
    
    drift_config = DriftConfig(
        drift_type=drift_type_map[drift_type_str],
        drift_magnitude=magnitude,
        affected_features=[0, 1, 2],
        drift_start_ratio=0.0,
        concept_drift=concept_drift,
    )
    
    X_stream, y_stream, _ = generator.generate_stream(
        n_batches * batch_size, drift_config
    )
    
    for i in range(n_batches):
        s = i * batch_size
        e = s + batch_size
        X_batch = X_stream.iloc[s:e].reset_index(drop=True)
        y_batch = y_stream.iloc[s:e].reset_index(drop=True)
        
        result = pipeline.process_batch(X_batch, y_batch)
        
        if result.get("status") != "buffering":
            st.session_state.batch_history.append({
                "batch_id": len(st.session_state.batch_history),
                "timestamp": pd.Timestamp.now(),
                "drift_score": result["drift_score"],
                "severity": result["severity"],
                "n_drifted_features": len(result["drifted_features"]),
                "drifted_features": result["drifted_features"],
                "model_f1": result.get("model_performance", {}).get("f1", None),
                "drift_type": drift_type_str,
                "magnitude": magnitude,
            })
            
            log_event(
                f"Batch {len(st.session_state.batch_history)}: "
                f"skor={result['drift_score']:.3f} [{result['severity']}]",
                "drift" if result['severity'] in ['high', 'critical'] else "info"
            )
            
            if result.get("mitigation") and result["mitigation"]["result"] != "skipped":
                log_event(
                    f"Müdahale: {result['mitigation']['actions']}",
                    "warn"
                )


def process_batches_csv(n_batches, batch_size):
    """CSV stream'inden batch'leri işle."""
    pipeline = st.session_state.pipeline
    X_stream, y_stream = st.session_state.stream_data
    
    for i in range(n_batches):
        start = st.session_state.stream_idx
        end = min(start + batch_size, len(X_stream))
        
        if start >= len(X_stream):
            log_event("Stream sonuna ulaşıldı", "warn")
            break
        
        X_batch = X_stream.iloc[start:end].reset_index(drop=True)
        y_batch = y_stream.iloc[start:end].reset_index(drop=True)
        
        result = pipeline.process_batch(X_batch, y_batch)
        st.session_state.stream_idx = end
        
        if result.get("status") != "buffering":
            st.session_state.batch_history.append({
                "batch_id": len(st.session_state.batch_history),
                "timestamp": pd.Timestamp.now(),
                "drift_score": result["drift_score"],
                "severity": result["severity"],
                "n_drifted_features": len(result["drifted_features"]),
                "drifted_features": result["drifted_features"],
                "model_f1": result.get("model_performance", {}).get("f1", None),
                "drift_type": "real_data",
                "magnitude": None,
            })
            
            log_event(
                f"Batch {len(st.session_state.batch_history)}: "
                f"skor={result['drift_score']:.3f} [{result['severity']}]",
                "drift" if result['severity'] in ['high', 'critical'] else "info"
            )


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — KONTROL PANELİ
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🎯 Drift Pipeline")
    st.caption("Real-Time Detection & Mitigation")
    
    st.divider()
    
    # ── Veri Modu Seçimi ─────────────────────────────────────
    st.subheader("📊 Veri Kaynağı")
    
    data_mode = st.radio(
        "Mod seç:",
        options=["synthetic", "csv_upload", "benchmark"],
        format_func=lambda x: {
            "synthetic": "🧪 Sentetik (kontrollü drift)",
            "csv_upload": "📁 Kendi CSV'mi yükle",
            "benchmark": "📚 Benchmark veri seti"
        }[x],
        index=0,
    )
    
    if data_mode != st.session_state.data_mode:
        st.session_state.data_mode = data_mode
        reset_pipeline()
    
    st.divider()
    
    # ── Pipeline Başlatma ────────────────────────────────────
    st.subheader("🚀 Pipeline Kontrol")
    
    # ── MOD 1: Sentetik Veri ─────────────────────────────────
    if data_mode == "synthetic":
        n_ref = st.slider("Referans veri boyutu", 1000, 5000, 3000, 500)
        seed = st.number_input("Random seed", value=42, step=1)
        
        if st.button("🎬 Pipeline'ı Başlat", type="primary", use_container_width=True):
            with st.spinner("Model eğitiliyor..."):
                generator, X_ref, y_ref = prepare_synthetic_data(n_ref, seed)
                pipeline = DriftPipeline()
                pipeline.model_manager.config.n_estimators = 50
                pipeline.model_manager.config.cv_folds = 3
                metrics = pipeline.initialize(X_ref, y_ref)
                
                st.session_state.pipeline = pipeline
                st.session_state.generator = generator
                st.session_state.reference_data = X_ref
                st.session_state.batch_history = []
                st.session_state.event_log = []
                
                log_event(f"Pipeline başlatıldı. Model F1: {metrics['f1']:.3f}", "success")
            st.rerun()
    
    # ── MOD 2: CSV Upload ────────────────────────────────────
    elif data_mode == "csv_upload":
        st.info("📌 CSV dosyan sayısal kolonlar ve 1 tane target kolon içermeli.")
        
        uploaded = st.file_uploader("CSV yükle", type=["csv"])
        
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.write(f"✅ Yüklendi: {len(df)} satır × {len(df.columns)} kolon")
                
                target_col = st.selectbox(
                    "Hedef (target) kolonu seç:",
                    options=df.columns.tolist(),
                )
                
                ref_ratio = st.slider(
                    "Referans oranı",
                    0.1, 0.5, 0.2, 0.05,
                    help="Küçük tut! İlk %20 referans = 'normal dönem', kalan %80 stream = 'değişim dönemi'"
                )
                st.caption(f"📌 Referans: ilk {int(len(df)*ref_ratio):,} satır | Stream: son {len(df)-int(len(df)*ref_ratio):,} satır")
                
                # Sayısal olmayan kolonları kontrol et
                non_numeric = [c for c in df.columns if c != target_col 
                              and not pd.api.types.is_numeric_dtype(df[c])]
                if non_numeric:
                    st.warning(f"⚠️ Sayısal olmayan kolonlar atlanacak: {non_numeric}")
                
                if st.button("🎬 Pipeline'ı Başlat", type="primary", use_container_width=True):
                    with st.spinner("Veri hazırlanıyor..."):
                        try:
                            # Sadece sayısal kolonları al
                            df_clean = df.select_dtypes(include=[np.number]).copy()
                            if target_col not in df_clean.columns:
                                df_clean[target_col] = df[target_col]
                            
                            # NaN'leri temizle
                            df_clean = df_clean.dropna()
                            
                            # Target sütununu integer'a çevir (0/1)
                            # Ondalıklı (0.0, 1.0) veya string ("UP"/"DOWN") olabilir
                            target_vals = df_clean[target_col]
                            unique_vals = target_vals.unique()
                            
                            if len(unique_vals) > 2:
                                # Çok sınıflı veya sürekli — median'a göre ikiye böl
                                median_val = target_vals.median()
                                df_clean[target_col] = (target_vals > median_val).astype(int)
                                st.info(f"ℹ️ Hedef sütun ikili sınıfa dönüştürüldü (median={median_val:.2f})")
                            else:
                                # Zaten ikili — sadece integer yap
                                min_val = target_vals.min()
                                df_clean[target_col] = (target_vals > min_val).astype(int)
                            
                            X_ref, y_ref, X_stream, y_stream = prepare_csv_data(
                                df_clean, target_col, ref_ratio
                            )
                            
                            # Son kontrol: y değerleri 0 ve 1 mi?
                            if len(y_ref.unique()) < 2:
                                st.error("❌ Hedef sütun tek değer içeriyor, farklı bir sütun seç.")
                                st.stop()
                            
                            pipeline = DriftPipeline()
                            pipeline.model_manager.config.n_estimators = 50
                            pipeline.model_manager.config.cv_folds = 3
                            metrics = pipeline.initialize(X_ref, y_ref)
                            
                            st.session_state.pipeline = pipeline
                            st.session_state.reference_data = X_ref
                            st.session_state.stream_data = (X_stream, y_stream)
                            st.session_state.batch_history = []
                            st.session_state.event_log = []
                            
                            log_event(f"CSV yüklendi. Model F1: {metrics['f1']:.3f}", "success")
                            log_event(f"Referans: {len(X_ref)}, Stream: {len(X_stream)}", "info")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Hata: {e}")
            except Exception as e:
                st.error(f"❌ CSV okuma hatası: {e}")
    
    # ── MOD 3: Benchmark ─────────────────────────────────────
    elif data_mode == "benchmark":
        st.info("📌 Drift araştırmalarında kullanılan standart veri setleri.")
        
        benchmark = st.selectbox(
            "Veri seti:",
            options=["electricity_synthetic"],
            format_func=lambda x: {
                "electricity_synthetic": "⚡ Electricity-like (simüle)"
            }[x],
        )
        
        st.caption("💡 Gerçek UCI Electricity veri seti indirildiğinde buraya eklenecek.")
        
        if st.button("🎬 Pipeline'ı Başlat", type="primary", use_container_width=True):
            # Electricity-benzeri sentetik veri (zamansal drift simülasyonu)
            with st.spinner("Benchmark veri hazırlanıyor..."):
                gen = DataGenerator(n_features=8, random_state=123)
                X_ref, y_ref = gen.generate_reference(n_samples=3000)
                
                pipeline = DriftPipeline()
                pipeline.model_manager.config.n_estimators = 50
                pipeline.model_manager.config.cv_folds = 3
                metrics = pipeline.initialize(X_ref, y_ref)
                
                st.session_state.pipeline = pipeline
                st.session_state.generator = gen
                st.session_state.reference_data = X_ref
                st.session_state.batch_history = []
                st.session_state.event_log = []
                
                log_event(f"Benchmark yüklendi. Model F1: {metrics['f1']:.3f}", "success")
            st.rerun()
    
    # ── Drift Enjeksiyon Kontrolleri ─────────────────────────
    if st.session_state.pipeline is not None:
        st.divider()
        st.subheader("💉 Drift Enjeksiyon")
        
        if data_mode == "synthetic" or data_mode == "benchmark":
            drift_type_str = st.selectbox(
                "Drift tipi:",
                options=["none", "sudden", "gradual", "incremental", "recurring"],
                format_func=lambda x: {
                    "none": "🟢 Drift yok (normal)",
                    "sudden": "⚡ Ani (sudden)",
                    "gradual": "🌊 Kademeli (gradual)",
                    "incremental": "📈 Artımlı (incremental)",
                    "recurring": "🔄 Tekrarlayan (recurring)",
                }[x],
            )
            
            # Drift yok seçiliyse şiddet ve concept drift gizle
            if drift_type_str == "none":
                magnitude = 0.0
                concept_drift = False
                st.info("ℹ️ Normal veri akışı — drift enjekte edilmiyor.")
            else:
                magnitude = st.slider("Drift şiddeti", 0.1, 1.0, 0.7, 0.1,
                                     help="0.1 = hafif drift, 1.0 = çok şiddetli")
                concept_drift = st.checkbox("Concept drift ekle", value=False,
                                           help="P(y|x) değişikliği — modelin karar mantığı da bozulsun")
            
            n_batches = st.slider("Kaç batch üret?", 1, 20, 5)
            batch_size = 300
            
            if st.button("▶️ Batch'leri Çalıştır", use_container_width=True):
                process_batches_synthetic(drift_type_str, magnitude, concept_drift, 
                                         n_batches, batch_size)
                st.rerun()
        
        elif data_mode == "csv_upload":
            if st.session_state.stream_data is not None:
                X_stream, y_stream = st.session_state.stream_data
                total = len(X_stream)
                processed = st.session_state.stream_idx
                remaining = total - processed
                progress = processed / total if total > 0 else 0
                
                # İlerleme göstergesi
                st.progress(progress, text=f"%{int(progress*100)} işlendi")
                col_a, col_b = st.columns(2)
                col_a.metric("İşlenen", f"{processed:,}")
                col_b.metric("Kalan", f"{remaining:,}")
                
                # Hangi dönemde olduğunu göster
                pct = processed / total * 100
                if pct < 30:
                    st.info("📍 Erken dönem — drift henüz yok")
                elif pct < 55:
                    st.warning("📍 Orta dönem — drift başlıyor")
                else:
                    st.error("📍 Geç dönem — güçlü drift bölgesi")
                
                batch_size = st.slider("Batch boyutu", 100, 1000, 300, 100)
                n_batches = st.slider("Kaç batch işle?", 1, 20, 5)
                
                if st.button("▶️ Batch'leri İşle", use_container_width=True,
                            disabled=(remaining <= 0)):
                    process_batches_csv(n_batches, batch_size)
                    st.rerun()
                
                if remaining <= 0:
                    st.success("✅ Tüm stream işlendi!")
        
        st.divider()
        
        if st.button("🔄 Pipeline'ı Sıfırla", use_container_width=True):
            reset_pipeline()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# ANA EKRAN
# ═══════════════════════════════════════════════════════════════════

st.title("📊 Real-Time Drift Detection Dashboard")

if st.session_state.pipeline is None:
    st.info("👈 Sol panelden bir veri modu seçip pipeline'ı başlat.")
    
    st.markdown("""
    ### Nasıl Kullanılır?
    
    1. **Veri modu seç** (sol panel):
       - 🧪 **Sentetik**: Kontrollü drift enjekte et, algoritmaları test et
       - 📁 **CSV Upload**: Kendi verini yükle, doğal drift var mı kontrol et
       - 📚 **Benchmark**: Standart drift veri setlerinden birini kullan
    
    2. **Pipeline'ı başlat**: Model otomatik eğitilir (~10 saniye)
    
    3. **Drift enjekte et** veya **stream'i işle**: Batch batch veri akışı başlar
    
    4. **Sonuçları izle**: Drift skoru, model performansı, otomatik müdahaleler
    """)
    
    st.stop()


# ── Pipeline aktifken gösterilecek panel ─────────────────────
pipeline = st.session_state.pipeline
state = pipeline.get_state()
history = st.session_state.batch_history

# ── ÜST METRIK KARTLARI ──────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    current_score = state["current_drift_score"]
    st.metric(
        "🎯 Drift Skoru",
        f"{current_score:.3f}",
        delta=None,
    )

with col2:
    severity = state["current_severity"]
    emoji = get_severity_emoji(severity)
    st.metric(
        "⚠️ Severity",
        f"{emoji} {severity.upper()}"
    )

with col3:
    model_f1 = state["model_metrics"].get("f1", 0)
    st.metric(
        "📈 Model F1",
        f"{model_f1:.3f}"
    )

with col4:
    st.metric(
        "🔄 Retrains",
        state["total_retrains"]
    )

with col5:
    st.metric(
        "📦 Batches",
        state["batches_processed"]
    )

st.divider()

# ── ANA GRAFİKLER ────────────────────────────────────────────
if history:
    df_hist = pd.DataFrame(history)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Drift Skoru Trendi")
        
        # Severity renklerini batch bazında belirle
        severity_colors = [get_severity_color(s) for s in df_hist["severity"]]
        
        fig = go.Figure()
        
        # Drift skoru çizgisi
        fig.add_trace(go.Scatter(
            x=df_hist["batch_id"],
            y=df_hist["drift_score"],
            mode="lines+markers",
            name="Drift Score",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=10, color=severity_colors, 
                       line=dict(color="white", width=1)),
            hovertemplate="Batch %{x}<br>Skor: %{y:.3f}<extra></extra>",
        ))
        
        # Threshold çizgileri
        fig.add_hline(y=0.30, line_dash="dash", line_color="#60a5fa", 
                     annotation_text="Low (0.30)", annotation_position="right")
        fig.add_hline(y=0.35, line_dash="dash", line_color="#fbbf24",
                     annotation_text="Medium (0.35)", annotation_position="right")
        fig.add_hline(y=0.55, line_dash="dash", line_color="#f87171",
                     annotation_text="High (0.55)", annotation_position="right")
        
        fig.update_layout(
            height=350,
            xaxis_title="Batch ID",
            yaxis_title="Drift Skoru",
            yaxis=dict(range=[0, 1]),
            template="plotly_dark",
            hovermode="x",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model F1 grafiği
        if df_hist["model_f1"].notna().any():
            st.subheader("🎯 Model Performansı (F1)")
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=df_hist["batch_id"],
                y=df_hist["model_f1"],
                mode="lines+markers",
                line=dict(color="#34d399", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(52, 211, 153, 0.1)",
            ))
            fig_f1.update_layout(
                height=250,
                xaxis_title="Batch ID",
                yaxis_title="F1 Score",
                yaxis=dict(range=[0, 1]),
                template="plotly_dark",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_f1, use_container_width=True)
    
    with col_right:
        st.subheader("📋 Event Log")
        if st.session_state.event_log:
            log_html = "<div class='event-log'>" + "<br>".join(
                st.session_state.event_log[:30]
            ) + "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.caption("Henüz event yok")
        
        st.subheader("📊 Severity Dağılımı")
        sev_counts = df_hist["severity"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=sev_counts.index,
            values=sev_counts.values,
            hole=0.5,
            marker=dict(colors=[get_severity_color(s) for s in sev_counts.index]),
        )])
        fig_pie.update_layout(
            height=250,
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # ── FEATURE DAĞILIM KARŞILAŞTIRMA ────────────────────────
    st.divider()
    st.subheader("🔬 Feature Dağılım Karşılaştırması")
    st.caption("Referans veri (mavi) vs son batch (kırmızı) — drift'in hangi feature'larda olduğunu gösterir.")
    
    if st.session_state.reference_data is not None:
        X_ref = st.session_state.reference_data
        
        # Son batch'in verisini al
        live_X, _ = pipeline.feature_store.get_live_window(100)
        
        if len(live_X) > 0:
            features_to_show = X_ref.columns[:6]  # İlk 6 feature
            
            fig_hist = make_subplots(
                rows=2, cols=3,
                subplot_titles=features_to_show,
            )
            
            for idx, feat in enumerate(features_to_show):
                row = idx // 3 + 1
                col = idx % 3 + 1
                
                fig_hist.add_trace(
                    go.Histogram(
                        x=X_ref[feat],
                        name="Referans",
                        opacity=0.6,
                        marker_color="#60a5fa",
                        showlegend=(idx == 0),
                        nbinsx=30,
                    ),
                    row=row, col=col,
                )
                fig_hist.add_trace(
                    go.Histogram(
                        x=live_X[feat],
                        name="Live",
                        opacity=0.6,
                        marker_color="#f87171",
                        showlegend=(idx == 0),
                        nbinsx=30,
                    ),
                    row=row, col=col,
                )
            
            fig_hist.update_layout(
                height=450,
                template="plotly_dark",
                barmode="overlay",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # ── BATCH DETAY TABLOSU ──────────────────────────────────
    st.divider()
    st.subheader("📋 Batch Geçmişi")
    
    display_df = df_hist.copy()
    display_df["drift_score"] = display_df["drift_score"].round(3)
    display_df["model_f1"] = display_df["model_f1"].round(3) if display_df["model_f1"].notna().any() else display_df["model_f1"]
    display_df["drifted_features"] = display_df["drifted_features"].apply(
        lambda x: ", ".join(x[:3]) + ("..." if len(x) > 3 else "") if x else "-"
    )
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
    
    st.dataframe(
        display_df[["batch_id", "timestamp", "drift_score", "severity", 
                   "n_drifted_features", "drifted_features", "model_f1"]],
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("▶️ Henüz batch işlenmedi. Sol panelden 'Batch'leri Çalıştır' butonuna bas.")