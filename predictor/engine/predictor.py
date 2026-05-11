"""
Predictor
=========
Egitilmis model ile yeni veri uzerinde tahmin yapar.

Baseline modeller:
  mean_baseline       — her zaman egitim ortalamasini tahmin et
  persistence_baseline— bir onceki degeri tahmin et (lag_1h varsa)

Her use case icin calisir:
  Classification: F1, AUC-ROC, accuracy
  Regression:     MAE, RMSE, R2
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from predictor.engine.types import TrainResult, PredictionResult
from predictor.engine.cleaner import DataCleaner


class Predictor:

    def _is_regression(self, y: np.ndarray) -> bool:
        return len(np.unique(y)) > 2 and y.max() > 10

    def _compute_metrics(self, y_true, y_pred, y_proba=None,
                          is_reg=False) -> Dict:
        if is_reg:
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2   = r2_score(y_true, y_pred)
            return {"mae": round(mae,2), "rmse": round(rmse,2),
                    "r2": round(r2,4), "f1": round(r2,4)}
        else:
            m = {
                "accuracy":  round(accuracy_score(y_true, y_pred), 4),
                "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            }
            if y_proba is not None:
                try:
                    m["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
                except: pass
            return m

    def predict(self,
                train_result: TrainResult,
                X_new: pd.DataFrame,
                y_new: Optional[pd.Series] = None) -> PredictionResult:

        X = DataCleaner.align_features(X_new, train_result.feature_cols)

        if train_result.scaler is not None:
            X_input = train_result.scaler.transform(X)
        else:
            X_input = X.values

        y_pred = train_result.model.predict(X_input)

        try:
            y_proba = train_result.model.predict_proba(X_input)[:, 1]
        except:
            y_proba = y_pred.astype(float)

        is_reg = self._is_regression(y_pred)
        predicted_rate = round(float(y_pred.mean()), 2) if is_reg \
                         else round(float(y_pred.mean()) * 100, 2)

        actual_rate = rate_error = metrics = None

        if y_new is not None and len(y_new) == len(y_pred):
            is_reg = self._is_regression(y_new.values)
            metrics = self._compute_metrics(y_new, y_pred, y_proba, is_reg)
            if is_reg:
                actual_rate    = round(float(y_new.mean()), 2)
                predicted_rate = round(float(y_pred.mean()), 2)
                rate_error     = round(float(metrics["mae"]), 2)
            else:
                actual_rate = round(float(y_new.mean()) * 100, 2)
                rate_error  = round(abs(actual_rate - predicted_rate), 2)

        return PredictionResult(
            algorithm=train_result.algorithm,
            predictions=y_pred,
            probabilities=y_proba,
            predicted_rate=predicted_rate,
            actual_rate=actual_rate,
            rate_error=rate_error,
            metrics=metrics,
            n_rows=len(y_pred),
        )

    def predict_all(self,
                    train_results: Dict[str, TrainResult],
                    X_new: pd.DataFrame,
                    y_new: Optional[pd.Series] = None) -> Dict[str, PredictionResult]:
        results = {}
        for algo, tr in train_results.items():
            try:
                results[algo] = self.predict(tr, X_new, y_new)
            except Exception as e:
                print(f"  [WARNING] {algo} prediction failed: {e}")
        return results

    def compute_baselines(self,
                          train_result: TrainResult,
                          X_new: pd.DataFrame,
                          y_new: pd.Series,
                          train_mean: float) -> Dict[str, PredictionResult]:
        """
        Baseline modelleri hesapla.

        mean_baseline:        Her zaman egitim ortalamasini tahmin et
        persistence_baseline: lag_1h sutunu varsa bir onceki degeri kullan
        """
        baselines = {}
        is_reg = self._is_regression(y_new.values)
        n = len(y_new)

        # 1. Mean baseline
        y_mean = np.full(n, train_mean)
        metrics_mean = self._compute_metrics(y_new, y_mean, is_reg=is_reg)
        mae_mean = float(mean_absolute_error(y_new, y_mean))
        baselines["mean_baseline"] = PredictionResult(
            algorithm="mean_baseline",
            predictions=y_mean,
            probabilities=y_mean,
            predicted_rate=round(train_mean, 2),
            actual_rate=round(float(y_new.mean()), 2) if is_reg else round(float(y_new.mean())*100, 2),
            rate_error=round(mae_mean, 2) if is_reg else round(abs(train_mean*100 - float(y_new.mean())*100), 2),
            metrics=metrics_mean,
            n_rows=n,
        )

        # 2. Persistence baseline (lag_1h varsa)
        if "lag_1h" in X_new.columns:
            y_pers = X_new["lag_1h"].values
            metrics_pers = self._compute_metrics(y_new, y_pers, is_reg=is_reg)
            mae_pers = float(mean_absolute_error(y_new, y_pers))
            baselines["persistence_baseline"] = PredictionResult(
                algorithm="persistence_baseline",
                predictions=y_pers,
                probabilities=y_pers,
                predicted_rate=round(float(y_pers.mean()), 2),
                actual_rate=round(float(y_new.mean()), 2) if is_reg else round(float(y_new.mean())*100, 2),
                rate_error=round(mae_pers, 2),
                metrics=metrics_pers,
                n_rows=n,
            )

        return baselines