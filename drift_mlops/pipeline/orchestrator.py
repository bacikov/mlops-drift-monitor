"""
Pipeline Orchestrator: Ties all components together into a working MLOps pipeline.

Flow:
1. Initialize with reference data → train baseline model
2. Stream new data batches
3. For each batch: detect drift → score → mitigate if needed
4. Log everything for monitoring
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field

from drift_mlops.config.settings import (
    DRIFT_THRESHOLDS, MODEL_CONFIG, PIPELINE_CONFIG,
    DriftSeverity, MitigationAction
)
from drift_mlops.data.generator import DataGenerator, DriftConfig, DriftType
from drift_mlops.data.feature_store import FeatureStore
from drift_mlops.detection.statistical import StatisticalDriftDetector
from drift_mlops.detection.streaming import StreamingDriftManager
from drift_mlops.detection.scorer import DriftScorer, DriftReport
from drift_mlops.mitigation.engine import MitigationEngine, ModelRetrainer, FallbackManager
from drift_mlops.models.model_manager import ModelManager


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    status: str = "initialized"  # initialized, running, paused, stopped
    batches_processed: int = 0
    total_drifts_detected: int = 0
    total_retrains: int = 0
    current_model_version: int = 0
    current_severity: str = "none"
    current_drift_score: float = 0.0
    model_metrics: Dict = field(default_factory=dict)
    last_update: Optional[str] = None


class DriftPipeline:
    """
    Main orchestrator for the drift detection & mitigation pipeline.
    
    Usage:
        pipeline = DriftPipeline()
        pipeline.initialize(reference_data_X, reference_data_y)
        
        for batch in data_stream:
            result = pipeline.process_batch(batch_X, batch_y)
            print(result)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Core components
        self.feature_store = FeatureStore()
        self.stat_detector = StatisticalDriftDetector()
        self.drift_scorer = DriftScorer()
        self.model_manager = ModelManager()
        self.mitigation_engine = MitigationEngine()
        self.fallback_manager = FallbackManager()
        self.retrainer = None  # initialized after first model train
        
        # Streaming detectors (initialized when we know features)
        self.stream_manager: Optional[StreamingDriftManager] = None
        
        # State
        self.state = PipelineState()
        self.drift_history: List[DriftReport] = []
        self.event_log: List[Dict] = []
        
        # Callbacks
        self._on_drift_detected: Optional[Callable] = None
        self._on_retrain: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None
    
    def initialize(self, X_ref: pd.DataFrame, y_ref: pd.Series):
        """
        Initialize the pipeline with reference data.
        Trains the baseline model and sets up all components.
        """
        self._log_event("pipeline_init", "Initializing pipeline...")
        
        # 1. Set reference data
        self.feature_store.set_reference(X_ref, y_ref)
        
        # 2. Train baseline model
        metrics = self.model_manager.train(X_ref, y_ref)
        self._log_event("model_trained", f"Baseline model v{self.model_manager.version}", metrics)
        
        # 3. Set up fallback
        self.fallback_manager.set_champion(
            self.model_manager.model, metrics.get("f1", 0)
        )
        
        # 4. Set up retrainer
        from sklearn.base import clone
        self.retrainer = ModelRetrainer(clone(self.model_manager.model))
        
        # 5. Initialize streaming detectors
        self.stream_manager = StreamingDriftManager(
            feature_names=list(X_ref.columns),
            adwin_delta=DRIFT_THRESHOLDS.adwin_delta,
            ph_threshold=DRIFT_THRESHOLDS.ph_threshold,
        )
        
        # 6. Register mitigation handlers
        self._register_handlers()
        
        # Update state
        self.state.status = "running"
        self.state.current_model_version = self.model_manager.version
        self.state.model_metrics = metrics
        self.state.last_update = datetime.now().isoformat()
        
        self._log_event("pipeline_ready", "Pipeline initialized and ready.")
        return metrics
    
    def process_batch(
        self,
        X_batch: pd.DataFrame,
        y_batch: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Process a single batch of data through the pipeline.
        
        Returns a dict with drift report, actions taken, and model status.
        """
        batch_start = time.time()
        
        # 1. Add to feature store
        self.feature_store.add_live_data(X_batch, y_batch)
        
        # 2. Get analysis window — sabit 500 örnek
        X_live, y_live = self.feature_store.get_live_window(500)
        
        if len(X_live) < 300:
            return {"status": "buffering", "buffer_size": len(X_live)}
        
        # 3. Statistical drift detection with configured thresholds
        thresholds_dict = {
            "ks_p_value": DRIFT_THRESHOLDS.ks_p_value,
            "psi_threshold": DRIFT_THRESHOLDS.psi_medium,
            "kl_threshold": DRIFT_THRESHOLDS.kl_threshold,
            "js_threshold": DRIFT_THRESHOLDS.js_threshold,
            "chi2_p_value": DRIFT_THRESHOLDS.chi2_p_value,
            "wasserstein_threshold": DRIFT_THRESHOLDS.wasserstein_threshold,
        }
        stat_results = self.stat_detector.detect_drift(
            self.feature_store.reference_data, X_live,
            thresholds=thresholds_dict
        )
        
        # 4. Streaming drift detection
        stream_results = {}
        for idx in range(len(X_batch)):
            row = X_batch.iloc[idx]
            feature_values = {col: row[col] for col in X_batch.columns}
            stream_results = self.stream_manager.update_features(feature_values)
            
            # DDM: check prediction errors
            if y_batch is not None and self.model_manager.is_fitted:
                pred = self.model_manager.predict(X_batch.iloc[[idx]])[0]
                is_error = pred != y_batch.iloc[idx]
                self.stream_manager.update_prediction(is_error)
        
        # 5. Compute drift score
        drift_report = self.drift_scorer.score(stat_results, stream_results)
        self.drift_history.append(drift_report)
        
        # 6. Execute mitigation if needed
        mitigation_record = None
        if drift_report.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            self.state.total_drifts_detected += 1
            
            mitigation_record = self.mitigation_engine.execute(
                drift_report=drift_report,
                model=self.model_manager.model,
                X_train=self.feature_store.reference_data,
                y_train=self.feature_store.reference_target,
                X_new=X_live,
                y_new=y_live,
            )
            
            self._log_event("mitigation_executed", 
                f"Severity: {drift_report.severity.value}",
                mitigation_record.to_dict())
            
            if self._on_drift_detected:
                self._on_drift_detected(drift_report)
        
        # 7. Evaluate current model performance
        model_perf = None
        if y_batch is not None and self.model_manager.is_fitted:
            model_perf = self.model_manager.evaluate(X_batch, y_batch)
        
        # Update state
        self.state.batches_processed += 1
        self.state.current_severity = drift_report.severity.value
        self.state.current_drift_score = drift_report.composite_score
        self.state.last_update = datetime.now().isoformat()
        if model_perf:
            self.state.model_metrics = model_perf
        
        elapsed = time.time() - batch_start
        
        return {
            "batch_id": self.state.batches_processed,
            "drift_score": round(drift_report.composite_score, 4),
            "severity": drift_report.severity.value,
            "drifted_features": drift_report.drifted_features,
            "n_tests_drifted": drift_report.n_tests_drifted,
            "n_tests_total": drift_report.n_tests_total,
            "model_performance": model_perf,
            "mitigation": mitigation_record.to_dict() if mitigation_record else None,
            "processing_time_ms": round(elapsed * 1000, 1),
        }
    
    def _register_handlers(self):
        """Register mitigation action handlers."""
        
        def handle_retrain(drift_report, model, X_train, y_train, X_new, y_new, **kwargs):
            if X_new is None or y_new is None or len(X_new) < 50:
                return {"skipped": "insufficient new data"}
            
            # Modeli yeni veriyle eğit
            X_combined = pd.concat([X_train, X_new], ignore_index=True)
            y_combined = pd.concat([y_train, y_new], ignore_index=True)
            
            metrics = self.model_manager.train(X_combined, y_combined)
            self.state.total_retrains += 1
            self.state.current_model_version = self.model_manager.version
            
            # Retrain başarılıysa referansı güncelle — bu mitigation'ın tamamlanması
            # "Kalıcı değişimi kabul et, yeni normali öğren" anlamına gelir
            # F1 makul seviyedeyse güncelle, değilse eski referansı koru
            new_f1 = metrics.get("f1", 0)
            reference_updated = new_f1 > 0.45
            if reference_updated:
                self.feature_store.update_reference(X_combined, y_combined)
            
            if self._on_retrain:
                self._on_retrain(metrics)
            
            return {
                "retrained": True, 
                "metrics": metrics,
                "reference_updated": reference_updated,
                "new_f1": new_f1,
            }
        
        def handle_alert(drift_report, **kwargs):
            alert = {
                "level": drift_report.severity.value,
                "score": drift_report.composite_score,
                "features": drift_report.drifted_features,
                "time": datetime.now().isoformat(),
            }
            if self._on_alert:
                self._on_alert(alert)
            return alert
        
        def handle_fallback(drift_report, **kwargs):
            switched = self.fallback_manager.switch_to_challenger()
            return {"switched": switched is not None}
        
        self.mitigation_engine.register_handler(MitigationAction.FULL_RETRAIN, handle_retrain)
        self.mitigation_engine.register_handler(MitigationAction.INCREMENTAL_RETRAIN, handle_retrain)
        self.mitigation_engine.register_handler(MitigationAction.ALERT, handle_alert)
        self.mitigation_engine.register_handler(MitigationAction.FALLBACK_MODEL, handle_fallback)
    
    def _log_event(self, event_type: str, message: str, data: Dict = None):
        """Log a pipeline event."""
        self.event_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "message": message,
            "data": data,
        })
    
    # ── Callback Registration ────────────────────────────────────────
    def on_drift(self, callback: Callable):
        self._on_drift_detected = callback
    
    def on_retrain(self, callback: Callable):
        self._on_retrain = callback
    
    def on_alert(self, callback: Callable):
        self._on_alert = callback
    
    # ── Convenience Methods ──────────────────────────────────────────
    def get_state(self) -> Dict:
        return {
            "status": self.state.status,
            "batches_processed": self.state.batches_processed,
            "total_drifts": self.state.total_drifts_detected,
            "total_retrains": self.state.total_retrains,
            "model_version": self.state.current_model_version,
            "current_severity": self.state.current_severity,
            "current_drift_score": self.state.current_drift_score,
            "model_metrics": self.state.model_metrics,
        }
    
    def get_drift_trend(self, n_last: int = 20) -> List[Dict]:
        """Get recent drift score history."""
        recent = self.drift_history[-n_last:]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "score": round(r.composite_score, 4),
                "severity": r.severity.value,
                "n_drifted_features": len(r.drifted_features),
            }
            for r in recent
        ]