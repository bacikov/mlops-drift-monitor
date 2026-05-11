"""
Model Trainer
=============
3 algoritmayı paralel eğitir, karşılaştırır.

Algoritmalar:
  Random Forest      — tree ensemble, robust
  XGBoost            — gradient boosting, genelde en yüksek doğruluk
  Logistic Regression — linear, hızlı, drift'e en dayanıklı

Eğitim süreci:
  - Train/test split: %80/%20 (held-out test — data leakage yok)
  - Logistic Regression için StandardScaler uygulanır
  - Dengesiz veri için class_weight='balanced'
  - Champion: en yüksek F1 skoru olan algoritma
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from predictor.engine.types import TrainResult
from predictor.config import ALGORITHMS, MODEL_CFG


class ModelTrainer:
    """
    3 algoritmayı eğitir. Classification ve regression destekler.
    Task otomatik belirlenir — _task sütununa göre.
    """

    def __init__(self):
        self.cfg = MODEL_CFG

    def _build_model(self, algorithm: str, task: str = "classification"):
        """Algoritma instance'ı oluştur."""
        if task == "regression":
            if algorithm == "random_forest":
                return RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )
            elif algorithm == "xgboost":
                if not XGBOOST_AVAILABLE: return None
                return XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                    verbosity=0,
                )
            elif algorithm == "logistic_regression":
                return Ridge(
                    alpha=1.0,
                )
        else:
            if algorithm == "random_forest":
                return RandomForestClassifier(
                    n_estimators=self.cfg.rf_n_estimators,
                    random_state=self.cfg.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            elif algorithm == "xgboost":
                if not XGBOOST_AVAILABLE: return None
                return XGBClassifier(
                    n_estimators=self.cfg.xgb_n_estimators,
                    max_depth=self.cfg.xgb_max_depth,
                    learning_rate=self.cfg.xgb_learning_rate,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                    verbosity=0,
                )
            elif algorithm == "logistic_regression":
                return LogisticRegression(
                    max_iter=self.cfg.lr_max_iter,
                    random_state=self.cfg.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                )
        return None

    def _compute_metrics(self, model, X_test, y_test,
                         scaler=None, task="classification") -> Dict:
        """Test setinde metrikleri hesapla."""
        X = scaler.transform(X_test) if scaler else X_test
        yp = model.predict(X)

        if task == "regression":
            mae  = mean_absolute_error(y_test, yp)
            rmse = np.sqrt(mean_squared_error(y_test, yp))
            r2   = r2_score(y_test, yp)
            return {
                "mae":  round(mae, 2),
                "rmse": round(rmse, 2),
                "r2":   round(r2, 4),
                "f1":   round(r2, 4),  # app.py uyumu için
            }

        metrics = {
            "accuracy":  round(accuracy_score(y_test, yp), 4),
            "f1":        round(f1_score(y_test, yp, zero_division=0), 4),
            "precision": round(precision_score(y_test, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_test, yp, zero_division=0), 4),
        }
        try:
            proba = model.predict_proba(X)[:, 1]
            metrics["auc_roc"] = round(roc_auc_score(y_test, proba), 4)
        except Exception:
            pass
        return metrics

    def train_one(self, algorithm: str,
                  X: pd.DataFrame,
                  y: pd.Series,
                  task: str = "classification") -> Optional[TrainResult]:
        model = self._build_model(algorithm, task)
        if model is None:
            return None

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
                stratify=y if task == "classification" else None,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y,
                test_size=self.cfg.test_size,
                random_state=self.cfg.random_state,
            )

        scaler = None
        if algorithm == "logistic_regression":
            scaler = StandardScaler()
            X_tr_fit = scaler.fit_transform(X_tr)
            model.fit(X_tr_fit, y_tr)
        else:
            model.fit(X_tr, y_tr)

        metrics = self._compute_metrics(model, X_te, y_te, scaler, task)

        train_rate = round(float(y.mean()) * 100, 2)
        X_te_input = scaler.transform(X_te) if scaler else X_te
        y_pred_test = model.predict(X_te_input)
        predicted_rate = round(float(y_pred_test.mean()) * 100, 2)

        return TrainResult(
            algorithm=algorithm,
            metrics=metrics,
            model=model,
            scaler=scaler,
            feature_cols=list(X.columns),
            target_col=y.name or "target",
            trained_at=datetime.now().isoformat(),
            n_train=len(X_tr),
            n_test=len(X_te),
            train_rate=train_rate,
            predicted_rate=predicted_rate,
        )

    def train_all(self, X: pd.DataFrame,
                  y: pd.Series,
                  task: str = "classification") -> Dict[str, TrainResult]:
        """Tüm algoritmaları eğit."""
        results = {}
        for algo in ALGORITHMS:
            try:
                r = self.train_one(algo, X, y, task=task)
                if r is not None:
                    results[algo] = r
            except Exception as e:
                print(f"  [WARNING] {algo} training failed: {e}")
        return results

    def get_champion(self, results: Dict[str, TrainResult],
                     metric: str = "f1") -> str:
        """En iyi metriği alan algoritmayı döndür."""
        scores = {
            algo: r.metrics.get(metric, 0)
            for algo, r in results.items()
        }
        if not scores:
            raise ValueError("No trained models to compare.")
        # Regression için R² kullan (yüksek = iyi)
        # Classification için F1 kullan (yüksek = iyi)
        return max(scores, key=scores.get)