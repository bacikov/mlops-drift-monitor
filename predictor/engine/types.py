"""
Data Types
==========
Tüm dataclass'lar burada tanımlı.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class TrainResult:
    """Tek algoritmanın eğitim sonucu."""
    algorithm:      str
    metrics:        Dict          # accuracy, f1, precision, recall, auc_roc
    model:          object
    scaler:         Optional[object]
    feature_cols:   List[str]
    target_col:     str
    trained_at:     str
    n_train:        int
    n_test:         int
    train_rate:     float         # gerçek oran eğitim verisinde (%)
    predicted_rate: float         # model tahmin oranı test setinde (%)


@dataclass
class PredictionResult:
    """Tek algoritmanın tahmin sonucu."""
    algorithm:      str
    predictions:    np.ndarray    # 0/1 her satır için
    probabilities:  np.ndarray    # 0-1 olasılık
    predicted_rate: float         # tahmin edilen oran (%)
    actual_rate:    Optional[float]  # gerçek oran (%) — etiket varsa
    rate_error:     Optional[float]  # |gerçek - tahmin| puan
    metrics:        Optional[Dict]   # etiket varsa classification metrics
    n_rows:         int


@dataclass
class FeatureDriftDetail:
    """Tek feature için drift detayı."""
    feature:        str
    ks_stat:        float
    ks_drifted:     bool
    psi:            float
    psi_drifted:    bool
    wasserstein:    float
    wa_drifted:     bool
    composite_score: float
    is_drifted:     bool


@dataclass
class DriftResult:
    """Tüm drift analizi sonucu."""
    drift_score:      float
    severity:         str          # none/low/medium/high/critical
    drifted_features: List[str]
    feature_scores:   Dict[str, float]
    feature_details:  List[FeatureDriftDetail]
    rate_drift:       Optional[float]   # |ref_rate - new_rate| puan
    n_features_total: int
    n_features_drifted: int


@dataclass
class AnalysisRecord:
    """Geçmişe kaydedilen tek analiz."""
    timestamp:        str
    dataset_name:     str
    use_case:         str
    champion:         str
    drift_score:      float
    severity:         str
    ref_rate:         float
    predicted_rate:   float
    actual_rate:      Optional[float]
    rate_error:       Optional[float]
    f1:               Optional[float]
    auc_roc:          Optional[float]
    drifted_features: List[str]
    retrained:        bool = False
    model_version:    Optional[str] = None
    all_algo_rates:   Dict[str, float] = field(default_factory=dict)
    all_algo_f1s:     Dict[str, float] = field(default_factory=dict)
