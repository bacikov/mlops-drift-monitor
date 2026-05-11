"""
Model Manager: Handles model training, evaluation, serialization, and serving.
"""
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

from drift_mlops.config.settings import MODEL_CONFIG


class ModelManager:
    """
    Manages the ML model lifecycle: training, evaluation, versioning.
    """
    
    MODEL_REGISTRY = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }
    
    def __init__(self, config: Optional[object] = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.metrics_history: list = []
        self._version = 0
        self._model_dir = "model_artifacts"
    
    def _create_model(self) -> object:
        """Create a new model instance based on config."""
        model_class = self.MODEL_REGISTRY.get(self.config.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        params = {}
        if self.config.model_type in ("random_forest", "gradient_boosting"):
            params = {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "random_state": self.config.random_state,
            }
        elif self.config.model_type == "logistic_regression":
            params = {"random_state": self.config.random_state, "max_iter": 1000}
        
        return model_class(**params)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale: bool = True,
    ) -> Dict:
        """Train the model and return metrics."""
        self.model = self._create_model()
        
        if scale:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns, index=X.index
            )
        else:
            X_scaled = X
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y,
            cv=self.config.cv_folds, scoring="f1_weighted"
        )
        
        # Final fit on all data
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._version += 1
        
        # Compute training metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            "version": self._version,
            "timestamp": datetime.now().isoformat(),
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "cv_f1_mean": round(np.mean(cv_scores), 4),
            "cv_f1_std": round(np.std(cv_scores), 4),
            "n_samples": len(X),
        }
        
        if y_proba is not None:
            metrics["auc_roc"] = round(roc_auc_score(y, y_proba), 4)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def predict(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        
        if scale:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns, index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        
        if scale:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns, index=X.index
            )
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, scale: bool = True) -> Dict:
        """Evaluate model on new data."""
        y_pred = self.predict(X, scale=scale)
        
        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "n_samples": len(X),
        }
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.predict_proba(X, scale=scale)[:, 1]
            metrics["auc_roc"] = round(roc_auc_score(y, y_proba), 4)
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importances if available."""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.scaler.feature_names_in_,
                self.model.feature_importances_
            ))
        return None
    
    def save(self, path: Optional[str] = None):
        """Save model and scaler to disk."""
        save_dir = path or self._model_dir
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(save_dir, f"model_v{self._version}.pkl"))
        joblib.dump(self.scaler, os.path.join(save_dir, f"scaler_v{self._version}.pkl"))
    
    def load(self, version: int, path: Optional[str] = None):
        """Load a specific model version."""
        load_dir = path or self._model_dir
        self.model = joblib.load(os.path.join(load_dir, f"model_v{version}.pkl"))
        self.scaler = joblib.load(os.path.join(load_dir, f"scaler_v{version}.pkl"))
        self._version = version
        self.is_fitted = True
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def latest_metrics(self) -> Optional[Dict]:
        return self.metrics_history[-1] if self.metrics_history else None
