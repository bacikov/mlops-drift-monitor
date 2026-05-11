"""
Global configuration for the Drift Detection Tool.
"""
from dataclasses import dataclass
from enum import Enum


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MitigationAction(Enum):
    NO_ACTION = "no_action"
    LOG_ONLY = "log_only"
    ALERT = "alert"
    INCREMENTAL_RETRAIN = "incremental_retrain"
    FULL_RETRAIN = "full_retrain"
    FEATURE_UPDATE = "feature_update"
    FALLBACK_MODEL = "fallback_model"
    PIPELINE_HALT = "pipeline_halt"


@dataclass
class DriftThresholds:
    """
    Drift tespit eşikleri.

    KS Test:
      p-value < 0.05 → drift var
      Not: Büyük örneklemlerde p-value her zaman düşük çıkar.
      KS istatistiği (D) daha güvenilir bir effect size ölçüsüdür.

    PSI (Population Stability Index):
      Endüstri standardı yorumlama (Siddiqi, 2006):
        PSI < 0.10 → stabil
        PSI 0.10-0.20 → dikkat
        PSI > 0.20 → ciddi kayma
      Bu projede 0.20 eşik kullanılmaktadır.

    Wasserstein Distance:
      Normalize edilmiş form (referans std'ye bölünmüş).
      0.10 eşiği ampirik olarak hava durumu veri seti üzerinde belirlendi.
      Farklı domainlerde kalibre edilmesi önerilir.

    Composite Score Eşikleri:
      PSI yorumlama kılavuzu referans alınarak belirlendi,
      hava durumu veri seti üzerinde doğrulandı.
      none:     < 0.15
      low:      0.15 - 0.30
      medium:   0.30 - 0.50
      high:     0.50 - 0.70
      critical: > 0.70
    """
    # KS Test
    ks_p_value: float = 0.05

    # PSI - endüstri standardı eşik
    psi_low: float = 0.10
    psi_medium: float = 0.20

    # Wasserstein - normalize edilmiş
    wasserstein_threshold: float = 0.10

    # Geriye dönük uyumluluk için tutulanlar
    # (monitor.py ve orchestrator.py bu isimleri kullanıyor)
    kl_threshold: float = 0.10
    js_threshold: float = 0.15
    chi2_p_value: float = 0.05


@dataclass
class ModelConfig:
    """Model eğitim konfigürasyonu."""
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    cv_folds: int = 5
    min_samples_retrain: int = 500
    retrain_cooldown_seconds: int = 300
    performance_drop_threshold: float = 0.05


@dataclass
class PipelineConfig:
    """Pipeline konfigürasyonu."""
    reference_window_size: int = 5000
    analysis_window_size: int = 500
    sliding_step: int = 100
    batch_interval_seconds: float = 5.0
    max_buffer_size: int = 10000
    dashboard_refresh_seconds: int = 3
    alert_cooldown_seconds: int = 60
    max_drift_history: int = 1000
    max_model_versions: int = 10


# Global config instances
DRIFT_THRESHOLDS = DriftThresholds()
MODEL_CONFIG = ModelConfig()
PIPELINE_CONFIG = PipelineConfig()