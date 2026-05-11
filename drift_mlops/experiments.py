"""
Deneysel Değerlendirme Scripti
===============================
4 drift tipi × 3 şiddet seviyesi = 12 senaryo
Her senaryoda 6 istatistiksel algoritma + 3 streaming algoritma test edilir.

Çıktılar:
- Karşılaştırma tablosu (CSV)
- Tespit doğruluğu (TP, FP, TN, FN)
- Tespit gecikmesi (kaç batch sonra drift algılandı)
- Retrain öncesi/sonrası model performansı

Kullanım:
    python experiments.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

from drift_mlops.data.generator import DataGenerator, DriftConfig, DriftType
from drift_mlops.data.feature_store import FeatureStore
from drift_mlops.detection.statistical import StatisticalDriftDetector
from drift_mlops.detection.streaming import ADWIN, PageHinkley, DDM
from drift_mlops.detection.scorer import DriftScorer
from drift_mlops.models.model_manager import ModelManager
from drift_mlops.pipeline.orchestrator import DriftPipeline


# ══════════════════════════════════════════════════════════════════════
# DENEY KONFİGÜRASYONU
# ══════════════════════════════════════════════════════════════════════

DRIFT_TYPES = [
    DriftType.SUDDEN,
    DriftType.GRADUAL,
    DriftType.INCREMENTAL,
    DriftType.RECURRING,
]

MAGNITUDES = [0.3, 0.6, 0.9]  # düşük, orta, yüksek

STAT_TEST_NAMES = [
    "ks_test", "psi", "kl_divergence",
    "js_divergence", "chi_square", "wasserstein"
]

N_REFERENCE = 2000
N_STREAM = 1000
BATCH_SIZE = 100
N_BATCHES = N_STREAM // BATCH_SIZE  # 10 batch
DRIFT_START = 0.3  # Drift verinin %30'unda başlar


def run_single_experiment(
    drift_type: DriftType,
    magnitude: float,
    seed: int = 42
) -> Dict:
    """
    Tek bir deney senaryosu çalıştırır.
    
    Returns:
        Dict with all metrics for this scenario
    """
    gen = DataGenerator(n_features=10, random_state=seed)
    
    # 1. Referans veri üret
    X_ref, y_ref = gen.generate_reference(N_REFERENCE)
    
    # 2. Driftli stream üret
    drift_config = DriftConfig(
        drift_type=drift_type,
        drift_magnitude=magnitude,
        affected_features=[0, 1, 2],
        drift_start_ratio=DRIFT_START,
        concept_drift=(magnitude >= 0.6),  # yüksek şiddette concept drift de ekle
    )
    X_stream, y_stream, drift_labels = gen.generate_stream(N_STREAM, drift_config)
    
    # 3. Model eğit
    model_mgr = ModelManager()
    model_mgr.config.n_estimators = 50
    model_mgr.config.cv_folds = 3
    baseline_metrics = model_mgr.train(X_ref, y_ref)
    
    # 4. Batch'lere böl
    batches = []
    for i in range(N_BATCHES):
        s = i * BATCH_SIZE
        e = s + BATCH_SIZE
        batches.append({
            "X": X_stream.iloc[s:e].reset_index(drop=True),
            "y": y_stream.iloc[s:e].reset_index(drop=True),
            "has_drift": drift_labels[s:e].any(),
            "drift_ratio": drift_labels[s:e].mean(),
        })
    
    # ── İstatistiksel Testler ──────────────────────────────────────
    detector = StatisticalDriftDetector()
    stat_results = {name: {"detections": [], "first_detection_batch": None} for name in STAT_TEST_NAMES}
    
    for batch_idx, batch in enumerate(batches):
        all_tests = detector.detect_drift(X_ref, batch["X"])
        
        for feature, tests in all_tests.items():
            for test_result in tests:
                name = test_result.test_name
                if test_result.is_drifted:
                    stat_results[name]["detections"].append(batch_idx)
                    if stat_results[name]["first_detection_batch"] is None:
                        stat_results[name]["first_detection_batch"] = batch_idx
    
    # ── Streaming Dedektörler ──────────────────────────────────────
    # ADWIN
    adwin = ADWIN(delta=0.002)
    adwin_detections = []
    adwin_first = None
    for batch_idx, batch in enumerate(batches):
        for val in batch["X"].iloc[:, 0].values:  # ilk feature'ı izle
            if adwin.update(val):
                adwin_detections.append(batch_idx)
                if adwin_first is None:
                    adwin_first = batch_idx
                break
    
    # Page-Hinkley
    ph = PageHinkley(threshold=50.0)
    ph_detections = []
    ph_first = None
    for batch_idx, batch in enumerate(batches):
        for val in batch["X"].iloc[:, 0].values:
            if ph.update(val):
                ph_detections.append(batch_idx)
                if ph_first is None:
                    ph_first = batch_idx
                break
    
    # DDM (model error based)
    ddm = DDM()
    ddm_detections = []
    ddm_first = None
    for batch_idx, batch in enumerate(batches):
        preds = model_mgr.predict(batch["X"])
        for pred, actual in zip(preds, batch["y"].values):
            if ddm.update(pred != actual):
                ddm_detections.append(batch_idx)
                if ddm_first is None:
                    ddm_first = batch_idx
                break
    
    # ── Ground Truth ───────────────────────────────────────────────
    # Drift hangi batch'lerde var?
    drift_batches = [i for i, b in enumerate(batches) if b["has_drift"]]
    no_drift_batches = [i for i, b in enumerate(batches) if not b["has_drift"]]
    first_drift_batch = drift_batches[0] if drift_batches else None
    
    # ── Metrik Hesaplama ───────────────────────────────────────────
    def compute_detection_metrics(detections: List[int], first_det) -> Dict:
        """Her algoritma için TP, FP, FN, TN ve gecikme hesapla."""
        detected_set = set(detections)
        
        tp = len(detected_set & set(drift_batches))
        fp = len(detected_set & set(no_drift_batches))
        fn = len(set(drift_batches) - detected_set)
        tn = len(set(no_drift_batches) - detected_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Gecikme: ilk gerçek drift batch'i ile ilk tespit arasındaki fark
        latency = None
        if first_det is not None and first_drift_batch is not None:
            latency = max(0, first_det - first_drift_batch)
        
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "detection_latency": latency,
            "n_detections": len(detections),
        }
    
    # ── Model Performans Etkisi ────────────────────────────────────
    # Drift öncesi ve sonrası model performansı
    pre_drift_perf = None
    post_drift_perf = None
    
    if no_drift_batches:
        b = batches[no_drift_batches[0]]
        pre_drift_perf = model_mgr.evaluate(b["X"], b["y"])
    
    if drift_batches:
        b = batches[drift_batches[-1]]
        post_drift_perf = model_mgr.evaluate(b["X"], b["y"])
    
    # ── Pipeline ile Retrain Testi ─────────────────────────────────
    pipeline = DriftPipeline()
    pipeline.model_manager.config.n_estimators = 50
    pipeline.model_manager.config.cv_folds = 3
    pipeline.initialize(X_ref, y_ref)
    
    for batch in batches:
        pipeline.process_batch(batch["X"], batch["y"])
    
    post_retrain_state = pipeline.get_state()
    
    # ── Sonuçları Topla ────────────────────────────────────────────
    results = {
        "drift_type": drift_type.value,
        "magnitude": magnitude,
        "n_drift_batches": len(drift_batches),
        "n_clean_batches": len(no_drift_batches),
        "baseline_f1": baseline_metrics["f1"],
        "pre_drift_f1": pre_drift_perf.get("f1") if pre_drift_perf else None,
        "post_drift_f1": post_drift_perf.get("f1") if post_drift_perf else None,
        "post_retrain_f1": post_retrain_state["model_metrics"].get("f1"),
        "total_retrains": post_retrain_state["total_retrains"],
        "algorithms": {},
    }
    
    # İstatistiksel testler
    for name in STAT_TEST_NAMES:
        results["algorithms"][name] = compute_detection_metrics(
            stat_results[name]["detections"],
            stat_results[name]["first_detection_batch"]
        )
    
    # Streaming dedektörler
    results["algorithms"]["ADWIN"] = compute_detection_metrics(adwin_detections, adwin_first)
    results["algorithms"]["PageHinkley"] = compute_detection_metrics(ph_detections, ph_first)
    results["algorithms"]["DDM"] = compute_detection_metrics(ddm_detections, ddm_first)
    
    return results


def run_all_experiments() -> List[Dict]:
    """Tüm 12 senaryoyu çalıştır."""
    all_results = []
    total = len(DRIFT_TYPES) * len(MAGNITUDES)
    
    print(f"\n{'='*70}")
    print(f"  DENEYSEL DEĞERLENDİRME")
    print(f"  {len(DRIFT_TYPES)} drift tipi × {len(MAGNITUDES)} şiddet = {total} senaryo")
    print(f"  Her senaryo: {len(STAT_TEST_NAMES)} istatistiksel + 3 streaming = 9 algoritma")
    print(f"{'='*70}\n")
    
    for i, drift_type in enumerate(DRIFT_TYPES):
        for j, magnitude in enumerate(MAGNITUDES):
            idx = i * len(MAGNITUDES) + j + 1
            print(f"  [{idx:2d}/{total}] {drift_type.value:12s} | magnitude={magnitude} ...", end=" ", flush=True)
            
            result = run_single_experiment(drift_type, magnitude, seed=42 + idx)
            all_results.append(result)
            
            # Özet yazdır
            best_algo = max(
                result["algorithms"].items(),
                key=lambda x: x[1]["f1"]
            )
            print(f"En iyi: {best_algo[0]} (F1={best_algo[1]['f1']})")
    
    return all_results


def generate_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Sonuçları karşılaştırma tablosuna dönüştür."""
    rows = []
    
    for r in results:
        for algo_name, metrics in r["algorithms"].items():
            rows.append({
                "Drift Type": r["drift_type"],
                "Magnitude": r["magnitude"],
                "Algorithm": algo_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "Detection Latency": metrics["detection_latency"],
                "False Positives": metrics["fp"],
                "True Positives": metrics["tp"],
            })
    
    return pd.DataFrame(rows)


def generate_model_impact_table(results: List[Dict]) -> pd.DataFrame:
    """Model performans etkisi tablosu."""
    rows = []
    for r in results:
        f1_drop = None
        if r["pre_drift_f1"] and r["post_drift_f1"]:
            f1_drop = round(r["pre_drift_f1"] - r["post_drift_f1"], 4)
        
        f1_recovery = None
        if r["post_drift_f1"] and r["post_retrain_f1"]:
            f1_recovery = round(r["post_retrain_f1"] - r["post_drift_f1"], 4)
        
        rows.append({
            "Drift Type": r["drift_type"],
            "Magnitude": r["magnitude"],
            "Baseline F1": r["baseline_f1"],
            "Pre-Drift F1": r["pre_drift_f1"],
            "Post-Drift F1": r["post_drift_f1"],
            "F1 Drop": f1_drop,
            "Post-Retrain F1": r["post_retrain_f1"],
            "F1 Recovery": f1_recovery,
            "N Retrains": r["total_retrains"],
        })
    
    return pd.DataFrame(rows)


def generate_algorithm_ranking(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Algoritmaları genel performansa göre sırala."""
    ranking = comparison_df.groupby("Algorithm").agg({
        "F1 Score": "mean",
        "Precision": "mean",
        "Recall": "mean",
        "False Positives": "mean",
        "Detection Latency": "mean",
    }).round(3)
    
    ranking = ranking.sort_values("F1 Score", ascending=False)
    ranking.index.name = "Algorithm"
    return ranking


def print_results(results: List[Dict], comparison_df: pd.DataFrame, 
                  impact_df: pd.DataFrame, ranking_df: pd.DataFrame):
    """Sonuçları güzel formatta yazdır."""
    
    print(f"\n{'='*70}")
    print("  SONUÇ 1: ALGORİTMA KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("  SONUÇ 2: ALGORİTMA SIRALAMASI (Ortalama F1'e göre)")
    print(f"{'='*70}")
    print(ranking_df.to_string())
    
    print(f"\n{'='*70}")
    print("  SONUÇ 3: MODEL PERFORMANS ETKİSİ")
    print(f"{'='*70}")
    print(impact_df.to_string(index=False))
    
    # En iyi algoritma hangi senaryoda?
    print(f"\n{'='*70}")
    print("  SONUÇ 4: SENARYO BAZINDA EN İYİ ALGORİTMA")
    print(f"{'='*70}")
    for _, group in comparison_df.groupby(["Drift Type", "Magnitude"]):
        best = group.loc[group["F1 Score"].idxmax()]
        print(f"  {best['Drift Type']:12s} mag={best['Magnitude']} → "
              f"{best['Algorithm']:15s} (F1={best['F1 Score']:.3f})")


def main():
    """Ana deney fonksiyonu."""
    # Tüm deneyleri çalıştır
    results = run_all_experiments()
    
    # Tabloları oluştur
    comparison_df = generate_comparison_table(results)
    impact_df = generate_model_impact_table(results)
    ranking_df = generate_algorithm_ranking(comparison_df)
    
    # Sonuçları yazdır
    print_results(results, comparison_df, impact_df, ranking_df)
    
    # CSV olarak kaydet
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df.to_csv(f"{output_dir}/algorithm_comparison.csv", index=False)
    impact_df.to_csv(f"{output_dir}/model_impact.csv", index=False)
    ranking_df.to_csv(f"{output_dir}/algorithm_ranking.csv")
    
    print(f"\n  📁 Sonuçlar '{output_dir}/' klasörüne kaydedildi.")
    print(f"     - algorithm_comparison.csv")
    print(f"     - model_impact.csv")
    print(f"     - algorithm_ranking.csv")
    print(f"\n{'='*70}")
    print(f"  Deneyler tamamlandı! ✅")
    print(f"{'='*70}\n")
    
    return results, comparison_df, impact_df, ranking_df


if __name__ == "__main__":
    main()
