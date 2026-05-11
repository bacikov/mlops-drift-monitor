"""
Main entry point: Demonstrates the full drift detection & mitigation pipeline.

Run: python main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drift_mlops.data.generator import DataGenerator, DriftConfig, DriftType
from drift_mlops.pipeline.orchestrator import DriftPipeline


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_section(text: str):
    print(f"\n--- {text} ---")


def run_demo():
    """Run a complete pipeline demonstration."""
    
    print_header("REAL-TIME DATA DRIFT DETECTION & MITIGATION PIPELINE")
    print("Graduation Project Demo\n")
    
    # ── Step 1: Generate Reference Data ──────────────────────────────
    print_section("Step 1: Generating reference (baseline) data")
    generator = DataGenerator(n_features=10, random_state=42)
    X_ref, y_ref = generator.generate_reference(n_samples=5000)
    print(f"  Reference data: {X_ref.shape[0]} samples, {X_ref.shape[1]} features")
    print(f"  Target distribution: {dict(y_ref.value_counts())}")
    print(f"  Features: {list(X_ref.columns)}")
    
    # ── Step 2: Initialize Pipeline ──────────────────────────────────
    print_section("Step 2: Initializing pipeline & training baseline model")
    pipeline = DriftPipeline()
    
    # Register callbacks
    def on_drift(report):
        print(f"  ⚠️  DRIFT DETECTED! Score={report.composite_score:.4f}, "
              f"Severity={report.severity.value}, "
              f"Features={report.drifted_features}")
    
    def on_retrain(metrics):
        print(f"  🔄 MODEL RETRAINED! F1={metrics['f1']:.4f}, "
              f"AUC={metrics.get('auc_roc', 'N/A')}")
    
    def on_alert(alert):
        print(f"  🔔 ALERT: Level={alert['level']}, Score={alert['score']:.4f}")
    
    pipeline.on_drift(on_drift)
    pipeline.on_retrain(on_retrain)
    pipeline.on_alert(on_alert)
    
    init_metrics = pipeline.initialize(X_ref, y_ref)
    print(f"  Baseline model metrics:")
    print(f"    Accuracy: {init_metrics['accuracy']}")
    print(f"    F1 Score: {init_metrics['f1']}")
    print(f"    AUC-ROC:  {init_metrics.get('auc_roc', 'N/A')}")
    print(f"    CV F1:    {init_metrics['cv_f1_mean']} ± {init_metrics['cv_f1_std']}")
    
    # ── Step 3: Simulate Normal Data Stream ──────────────────────────
    print_section("Step 3: Processing NORMAL data stream (no drift)")
    normal_config = DriftConfig(drift_type=DriftType.NONE)
    normal_batches = generator.generate_batch_stream(
        n_batches=5, batch_size=200, drift_config=normal_config
    )
    
    for batch in normal_batches:
        result = pipeline.process_batch(batch["X"], batch["y"])
        if result.get("status") == "buffering":
            print(f"  Batch {batch['batch_id']}: Buffering ({result['buffer_size']} samples)")
        else:
            perf = result.get("model_performance", {})
            print(f"  Batch {batch['batch_id']}: "
                  f"Drift={result['drift_score']:.4f} [{result['severity']}] | "
                  f"F1={perf.get('f1', '?')}")
    
    # ── Step 4: Simulate Sudden Drift ────────────────────────────────
    print_section("Step 4: Injecting SUDDEN DRIFT (magnitude=0.8)")
    sudden_config = DriftConfig(
        drift_type=DriftType.SUDDEN,
        drift_magnitude=0.8,
        affected_features=[0, 1, 2],  # income, age, credit_score
        drift_start_ratio=0.0,  # drift from the start
        concept_drift=False,
    )
    drift_batches = generator.generate_batch_stream(
        n_batches=8, batch_size=200, drift_config=sudden_config
    )
    
    for batch in drift_batches:
        result = pipeline.process_batch(batch["X"], batch["y"])
        if result.get("status") == "buffering":
            continue
        perf = result.get("model_performance", {})
        mitigation = result.get("mitigation")
        mit_str = ""
        if mitigation and mitigation["result"] != "skipped":
            mit_str = f" | Actions: {mitigation['actions']}"
        print(f"  Batch {batch['batch_id']}: "
              f"Drift={result['drift_score']:.4f} [{result['severity']}] | "
              f"F1={perf.get('f1', '?')} | "
              f"Drifted={result['drifted_features']}{mit_str}")
    
    # ── Step 5: Simulate Gradual Drift ───────────────────────────────
    print_section("Step 5: Injecting GRADUAL DRIFT (magnitude=0.6)")
    gradual_config = DriftConfig(
        drift_type=DriftType.GRADUAL,
        drift_magnitude=0.6,
        affected_features=[3, 4, 5],  # debt_ratio, employment_years, num_accounts
        drift_start_ratio=0.2,
        concept_drift=True,  # also shift decision boundary
    )
    gradual_batches = generator.generate_batch_stream(
        n_batches=10, batch_size=200, drift_config=gradual_config
    )
    
    for batch in gradual_batches:
        result = pipeline.process_batch(batch["X"], batch["y"])
        if result.get("status") == "buffering":
            continue
        perf = result.get("model_performance", {})
        print(f"  Batch {batch['batch_id']}: "
              f"Drift={result['drift_score']:.4f} [{result['severity']}] | "
              f"F1={perf.get('f1', '?')} | "
              f"Drifted={result['drifted_features']}")
    
    # ── Step 6: Pipeline Summary ─────────────────────────────────────
    print_header("PIPELINE SUMMARY")
    state = pipeline.get_state()
    print(f"  Status:           {state['status']}")
    print(f"  Batches Processed:{state['batches_processed']}")
    print(f"  Drifts Detected:  {state['total_drifts']}")
    print(f"  Model Retrains:   {state['total_retrains']}")
    print(f"  Model Version:    v{state['model_version']}")
    print(f"  Current Severity: {state['current_severity']}")
    print(f"  Current Score:    {state['current_drift_score']:.4f}")
    
    print_section("Drift Score Trend (last 10)")
    trend = pipeline.get_drift_trend(10)
    for t in trend:
        bar = "█" * int(t["score"] * 40)
        print(f"  {t['severity']:>8} | {t['score']:.4f} | {bar}")
    
    print_section("Mitigation Action Summary")
    summary = pipeline.mitigation_engine.get_action_summary()
    print(f"  Total Records: {summary.get('total_records', 0)}")
    if 'action_counts' in summary:
        for action, count in summary['action_counts'].items():
            print(f"    {action}: {count}")
    
    print(f"\n{'='*70}")
    print("  Demo completed successfully! ✅")
    print(f"{'='*70}\n")
    
    return pipeline


if __name__ == "__main__":
    pipeline = run_demo()
