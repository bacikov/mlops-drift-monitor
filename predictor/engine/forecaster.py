"""
Rate Forecaster v2
==================
Expanding window CV + baseline modelleri eklendi.

Baseline modeller:
  persistence  — "Bu hafta gecen hafta gibi olur"
  mean         — "Her zaman egitim ortalamasi"
  seasonal_mean— "Bu haftanin tarihi ortalamasini kullan"

CV Yontemi:
  Expanding window (walk-forward validation)
  Zaman serisi icin dogru yontem — gelecegi tahmin etmek icin
  gelecek verisi KULLANILMAZ.

  Ornek (5 fold, 413 hafta):
    Fold 1: train 1-330, test 331-413
    Fold 2: train 1-347, test 348-413
    ...
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


FORECAST_ALGORITHMS = {
    "rf_regressor": {
        "label": "Random Forest",
        "desc":  "Tree ensemble — robust, nonlinear",
        "color": "#3b82f6",
    },
    "xgb_regressor": {
        "label": "XGBoost",
        "desc":  "Gradient boosting — usually most accurate",
        "color": "#f59e0b",
    },
    "linear": {
        "label": "Ridge Regression",
        "desc":  "Regularized linear — interpretable baseline",
        "color": "#22c55e",
    },
    # Baseline modeller
    "persistence": {
        "label": "Persistence Baseline",
        "desc":  "Predicts: this week = last week",
        "color": "#71717a",
    },
    "mean_baseline": {
        "label": "Mean Baseline",
        "desc":  "Predicts: always the training mean",
        "color": "#52525b",
    },
}


@dataclass
class ForecastTrainResult:
    algorithm:       str
    model:           object
    scaler:          Optional[object]
    feature_cols:    List[str]
    target_col:      str
    trained_at:      str
    n_train:         int
    mae:             float
    rmse:            float
    r2:              float
    cv_mae:          float
    cv_method:       str        # "expanding_window" veya "loo"
    train_mean_rate: float
    train_std_rate:  float
    is_baseline:     bool = False


@dataclass
class ForecastPrediction:
    algorithm:       str
    predicted_rates: np.ndarray
    actual_rates:    Optional[np.ndarray]
    mean_predicted:  float
    mean_actual:     Optional[float]
    mae:             Optional[float]
    rmse:            Optional[float]
    r2:              Optional[float]
    is_baseline:     bool = False


class RateForecaster:
    """
    Expanding window CV ile regression modelleri.
    Persistence ve mean baseline ile karsilastirma.
    """

    def __init__(self, random_state: int = 42, n_cv_folds: int = 5):
        self.random_state = random_state
        self.n_cv_folds   = n_cv_folds

    def _build(self, algo: str):
        if algo == "rf_regressor":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                random_state=self.random_state,
            )
        elif algo == "xgb_regressor":
            if not XGBOOST_AVAILABLE: return None
            return XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            )
        elif algo == "linear":
            return Ridge(alpha=1.0)
        return None

    def _metrics(self, y_true, y_pred) -> Dict:
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return {
            "mae":  round(mae * 100, 3),
            "rmse": round(rmse * 100, 3),
            "r2":   round(r2, 4),
        }

    def _expanding_window_cv(self, model_builder, X: np.ndarray,
                             y: np.ndarray, scaler=None) -> float:
        """
        Expanding window (walk-forward) cross-validation.

        Zaman serisi icin dogru CV yontemi.
        Her fold'da: gecmis = train, gelecek = test.
        Gelecek bilgisi kesinlikle train'e sizmiyor.
        """
        n = len(X)
        min_train = max(int(n * 0.6), 20)
        fold_size = max(int((n - min_train) / self.n_cv_folds), 1)

        fold_maes = []
        for fold in range(self.n_cv_folds):
            test_start = min_train + fold * fold_size
            test_end   = min(test_start + fold_size, n)
            if test_start >= n:
                break

            X_tr = X[:test_start]
            y_tr = y[:test_start]
            X_te = X[test_start:test_end]
            y_te = y[test_start:test_end]

            if len(X_tr) < 10 or len(X_te) == 0:
                continue

            m = model_builder()
            if scaler:
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                m.fit(X_tr_s, y_tr)
                y_pred = m.predict(X_te_s)
            else:
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)

            fold_maes.append(mean_absolute_error(y_te, y_pred))

        return round(float(np.mean(fold_maes)) * 100, 3) if fold_maes else 999.0

    def train_one(self, algo: str,
                  X: pd.DataFrame,
                  y: pd.Series) -> Optional[ForecastTrainResult]:
        model = self._build(algo)
        if model is None:
            return None

        n = len(X)
        scaler = None

        if algo == "linear":
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X)
        else:
            X_fit = X.values

        # Tam veriyle egit
        model.fit(X_fit, y.values)
        y_pred_train = model.predict(X_fit)
        m = self._metrics(y.values, y_pred_train)

        # Expanding window CV
        cv_mae = self._expanding_window_cv(
            lambda: self._build(algo),
            X_fit, y.values,
            scaler=scaler,
        )

        return ForecastTrainResult(
            algorithm=algo, model=model, scaler=scaler,
            feature_cols=list(X.columns),
            target_col=y.name or "delay_rate",
            trained_at=datetime.now().isoformat(),
            n_train=n,
            mae=m["mae"], rmse=m["rmse"], r2=m["r2"],
            cv_mae=cv_mae,
            cv_method="expanding_window",
            train_mean_rate=round(float(y.mean()) * 100, 2),
            train_std_rate=round(float(y.std()) * 100, 2),
            is_baseline=False,
        )

    def train_baselines(self, y: pd.Series) -> Dict[str, ForecastTrainResult]:
        """
        Persistence ve mean baseline'lari olustur.
        Bunlar "model olmadan ne kadar dogru tahmin yapilabilir?"
        sorusunu cevaplar.
        """
        baselines = {}
        train_mean = float(y.mean())

        # Persistence baseline — onceki hafta ne idiyse bu hafta o
        # Egitim verisi uzerinde: shift(1) ile hesapla
        y_persistence = y.shift(1).fillna(train_mean).values
        m_pers = self._metrics(y.values, y_persistence)

        baselines["persistence"] = ForecastTrainResult(
            algorithm="persistence", model=None, scaler=None,
            feature_cols=["prev_week_delay"],
            target_col=y.name or "delay_rate",
            trained_at=datetime.now().isoformat(),
            n_train=len(y),
            mae=m_pers["mae"], rmse=m_pers["rmse"], r2=m_pers["r2"],
            cv_mae=m_pers["mae"],
            cv_method="n/a",
            train_mean_rate=round(train_mean * 100, 2),
            train_std_rate=round(float(y.std()) * 100, 2),
            is_baseline=True,
        )

        # Mean baseline — her zaman ortalama tahmin et
        y_mean = np.full(len(y), train_mean)
        m_mean = self._metrics(y.values, y_mean)

        baselines["mean_baseline"] = ForecastTrainResult(
            algorithm="mean_baseline", model=None, scaler=None,
            feature_cols=[],
            target_col=y.name or "delay_rate",
            trained_at=datetime.now().isoformat(),
            n_train=len(y),
            mae=m_mean["mae"], rmse=m_mean["rmse"], r2=m_mean["r2"],
            cv_mae=m_mean["mae"],
            cv_method="n/a",
            train_mean_rate=round(train_mean * 100, 2),
            train_std_rate=round(float(y.std()) * 100, 2),
            is_baseline=True,
        )

        return baselines

    def train_all(self, X: pd.DataFrame,
                  y: pd.Series,
                  include_baselines: bool = True) -> Dict[str, ForecastTrainResult]:
        results = {}

        # ML modeller
        for algo in ["rf_regressor", "xgb_regressor", "linear"]:
            try:
                r = self.train_one(algo, X, y)
                if r is not None:
                    results[algo] = r
            except Exception as e:
                print(f"  [WARNING] {algo} failed: {e}")

        # Baseline modeller
        if include_baselines:
            baselines = self.train_baselines(y)
            results.update(baselines)

        return results

    def get_champion(self, results: Dict[str, ForecastTrainResult]) -> str:
        """
        Sadece ML modeller arasinda seciyor (baseline haric).
        Expanding window CV MAE'ye gore.
        """
        ml_only = {k: v for k, v in results.items() if not v.is_baseline}
        if not ml_only:
            return list(results.keys())[0]
        return min(ml_only, key=lambda k: ml_only[k].cv_mae)

    def predict_one(self, result: ForecastTrainResult,
                    X_new: pd.DataFrame,
                    y_new: Optional[pd.Series] = None) -> ForecastPrediction:

        # Baseline tahminleri
        if result.is_baseline:
            if result.algorithm == "persistence":
                # prev_week_delay sütununu kullan
                if "prev_week_delay" in X_new.columns:
                    y_pred = X_new["prev_week_delay"].values
                else:
                    y_pred = np.full(len(X_new), result.train_mean_rate / 100)
            else:  # mean_baseline
                y_pred = np.full(len(X_new), result.train_mean_rate / 100)

            y_pred = np.clip(y_pred, 0, 1)
            mean_predicted = round(float(y_pred.mean()) * 100, 2)
            mean_actual = mae = rmse = r2 = None

            if y_new is not None and len(y_new) == len(y_pred):
                mean_actual = round(float(y_new.mean()) * 100, 2)
                m = self._metrics(y_new.values, y_pred)
                mae = m["mae"]; rmse = m["rmse"]; r2 = m["r2"]

            return ForecastPrediction(
                algorithm=result.algorithm,
                predicted_rates=y_pred,
                actual_rates=y_new.values if y_new is not None else None,
                mean_predicted=mean_predicted, mean_actual=mean_actual,
                mae=mae, rmse=rmse, r2=r2,
                is_baseline=True,
            )

        # ML model tahmini
        missing = [c for c in result.feature_cols if c not in X_new.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X = X_new[result.feature_cols].copy()

        if result.scaler:
            X_input = result.scaler.transform(X)
        else:
            X_input = X.values

        y_pred = np.clip(result.model.predict(X_input), 0, 1)
        mean_predicted = round(float(y_pred.mean()) * 100, 2)
        mean_actual = mae = rmse = r2 = None

        if y_new is not None and len(y_new) == len(y_pred):
            mean_actual = round(float(y_new.mean()) * 100, 2)
            m = self._metrics(y_new.values, y_pred)
            mae = m["mae"]; rmse = m["rmse"]; r2 = m["r2"]

        return ForecastPrediction(
            algorithm=result.algorithm,
            predicted_rates=y_pred,
            actual_rates=y_new.values if y_new is not None else None,
            mean_predicted=mean_predicted, mean_actual=mean_actual,
            mae=mae, rmse=rmse, r2=r2,
            is_baseline=False,
        )

    def predict_all(self, results: Dict[str, ForecastTrainResult],
                    X_new: pd.DataFrame,
                    y_new: Optional[pd.Series] = None) -> Dict[str, ForecastPrediction]:
        predictions = {}
        for algo, r in results.items():
            try:
                predictions[algo] = self.predict_one(r, X_new, y_new)
            except Exception as e:
                print(f"  [WARNING] {algo} prediction failed: {e}")
        return predictions

    def get_feature_importance(self,
                               result: ForecastTrainResult) -> Optional[pd.DataFrame]:
        if result.is_baseline or not hasattr(result.model, "feature_importances_"):
            return None
        imp = result.model.feature_importances_
        return pd.DataFrame({
            "feature":    result.feature_cols,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    def ablation_test(self, X: pd.DataFrame, y: pd.Series,
                      feature_to_remove: str) -> Dict:
        """
        Tek bir feature'i cikartarak modelin performansini olc.
        prev_week_delay ne kadar katkı yapiyor?
        """
        if feature_to_remove not in X.columns:
            return {"error": f"{feature_to_remove} not in features"}

        X_reduced = X.drop(columns=[feature_to_remove])

        results_full    = self.train_all(X, y, include_baselines=False)
        results_reduced = self.train_all(X_reduced, y, include_baselines=False)

        comparison = {}
        for algo in results_full:
            if algo in results_reduced:
                full_mae    = results_full[algo].cv_mae
                reduced_mae = results_reduced[algo].cv_mae
                comparison[algo] = {
                    "full_cv_mae":    full_mae,
                    "reduced_cv_mae": reduced_mae,
                    "delta":          round(reduced_mae - full_mae, 3),
                    "feature_contribution_pp": round(reduced_mae - full_mae, 3),
                }

        return comparison