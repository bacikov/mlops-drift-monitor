"""
Predictor Configuration
========================
Tüm eşikler, sabitler ve ayarlar burada.
"""

from dataclasses import dataclass


# ── Algorithms ────────────────────────────────────────────────────

ALGORITHMS = {
    "random_forest": {
        "label": "Random Forest",
        "desc":  "Tree ensemble — robust, handles nonlinear patterns well",
        "color": "#3b82f6",
    },
    "xgboost": {
        "label": "XGBoost",
        "desc":  "Gradient boosting — usually highest accuracy",
        "color": "#f59e0b",
    },
    "logistic_regression": {
        "label": "Ridge Regression",
        "desc":  "Regularized linear model — fast, interpretable, drift-resistant baseline",
        "color": "#22c55e",
    },
}

# ── Use Cases ─────────────────────────────────────────────────────

USE_CASES = {
    "Bike Sharing": {
        "ref":    "bike_reference.csv",
        "early":  "bike_new_early.csv",
        "late":   "bike_new_late.csv",
        "target": "target",
        "event":  "bike rentals",
        "unit":   "bikes/hour",
        "task":   "regression",
        "desc":   "Predict hourly bike rental count. Winter vs summer drift — temperature, humidity, weather all change.",
    },
    "Energy Consumption": {
        "ref":    "energy_reference.csv",
        "early":  "energy_new_early.csv",
        "late":   "energy_new_late.csv",
        "target": "target",
        "event":  "energy consumption",
        "unit":   "MW",
        "task":   "regression",
        "desc":   "Predict hourly energy consumption in megawatts (PJM East region, 2002-2018).",
    },
    "Airline Delay": {
        "ref":    "airline_reference_sample.csv",
        "early":  "airline_new_early_sample.csv",
        "late":   "airline_new_late_sample.csv",
        "target": "target",
        "event":  "flight delay",
        "unit":   "%",
        "task":   "classification",
        "desc":   "Predict what percentage of flights will be delayed 15+ minutes.",
    },
    "Credit Default": {
        "ref":    "credit_reference.csv",
        "early":  "credit_new_early.csv",
        "late":   "credit_new_late.csv",
        "target": "target",
        "event":  "credit default",
        "unit":   "%",
        "task":   "classification",
        "desc":   "Predict what percentage of credit card holders will default next month.",
    },
    "Weather Rain": {
        "ref":    "weather_reference.csv",
        "early":  "weather_new_early.csv",
        "late":   "weather_new_late.csv",
        "target": "target",
        "event":  "rain tomorrow",
        "unit":   "%",
        "task":   "classification",
        "desc":   "Predict what percentage of days will have rain tomorrow.",
    },
    "Custom": {
        "ref":    None,
        "early":  None,
        "late":   None,
        "target": None,
        "event":  "positive outcome",
        "unit":   "",
        "task":   "auto",
        "desc":   "Upload any dataset.",
    },
}

# ── Drift Thresholds ──────────────────────────────────────────────

@dataclass
class DriftThresholds:
    """
    Drift eşik değerleri.

    KS p-value: Büyük örneklemlerde (n > 10.000) p-value her zaman
    küçük çıkar — bu yüzden p-value yerine istatistik (D) değeri
    normalize edilerek kullanılır.

    PSI: Bankacılık sektörü standardı (Siddiqi 2006):
      < 0.10 → stabil
      0.10-0.20 → dikkat
      > 0.20 → ciddi kayma

    Composite eşikler PSI standardına dayanarak belirlendi ve
    hava durumu + airline veri setlerinde ampirik olarak doğrulandı.
    Farklı domainlerde kalibre edilmesi önerilir.
    """
    # İstatistiksel testler
    ks_p_value:            float = 0.05
    psi_warning:           float = 0.10
    psi_critical:          float = 0.20
    wasserstein_threshold: float = 0.10

    # Normalizasyon üst sınırları
    ks_norm_max:   float = 0.30   # KS stat bu değerin üstü → tam drift
    psi_norm_max:  float = 0.40   # PSI bu değerin üstü → tam drift
    was_norm_max:  float = 0.30   # Wasserstein bu değerin üstü → tam drift

    # Feature drifted sayılma eşiği
    feature_drift_threshold: float = 0.50

    # Composite severity eşikleri
    severity_low:      float = 0.15
    severity_medium:   float = 0.30
    severity_high:     float = 0.50
    severity_critical: float = 0.70

    # Rate drift eşikleri (yüzde puan)
    rate_drift_warning:  float = 2.0
    rate_drift_critical: float = 5.0


# ── Model Config ──────────────────────────────────────────────────

@dataclass
class ModelConfig:
    test_size:    float = 0.20
    random_state: int   = 42
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth:    int = None
    # XGBoost
    xgb_n_estimators:  int   = 100
    xgb_max_depth:     int   = 6
    xgb_learning_rate: float = 0.1
    # Logistic Regression
    lr_max_iter: int = 1000


# ── Retrain Config ────────────────────────────────────────────────

@dataclass
class RetrainConfig:
    default_new_weight:    float = 0.70
    min_new_weight:        float = 0.10
    max_new_weight:        float = 0.90
    min_samples_retrain:   int   = 100
    f1_improvement_target: float = 0.02  # retrain başarılı sayılır eşiği


# ── Global instances ──────────────────────────────────────────────

DRIFT_CFG   = DriftThresholds()
MODEL_CFG   = ModelConfig()
RETRAIN_CFG = RetrainConfig()

# ── UI Colors ─────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "none":     "#22c55e",
    "low":      "#3b82f6",
    "medium":   "#f59e0b",
    "high":     "#ef4444",
    "critical": "#dc2626",
}

SEVERITY_CARD = {
    "none":     "green",
    "low":      "blue",
    "medium":   "yellow",
    "high":     "red",
    "critical": "red",
}