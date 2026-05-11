"""
Multi-Model Trainer
====================
3 farklı algoritma ile model eğitimi ve karşılaştırma.

Algoritmalar:
1. Random Forest    - Tree-based ensemble, robust, az tuning gerektirir
2. XGBoost          - Gradient boosting, genelde en yüksek doğruluk
3. Logistic Reg.    - Linear, hızlı, baseline ve yorumlanabilir

Kullanım:
    trainer = MultiModelTrainer()
    results = trainer.train_all(X, y)
    # results = {"random_forest": {model, metrics}, "xgboost": {...}, ...}

    best = trainer.get_champion(results)  # En iyi F1'e sahip model
"""

from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ALGORITHMS = {
    "random_forest": {
        "label": "Random Forest",
        "description": "Tree ensemble — robust, no tuning needed",
    },
    "xgboost": {
        "label": "XGBoost",
        "description": "Gradient boosting — usually highest accuracy",
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "description": "Linear baseline — fast, interpretable",
    },
}


class MultiModelTrainer:
    """3 algoritmayı paralel eğitir, metrikleri karşılaştırır."""

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.scalers = {}  # Logistic için scaler tutulacak

    def _build_model(self, algorithm: str):
        """Tek bir algoritma instance'ı oluştur."""
        if algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif algorithm == "xgboost":
            if not XGBOOST_AVAILABLE:
                return None
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        elif algorithm == "logistic_regression":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
            )
        return None

    def _evaluate(self, model, X_test, y_test) -> Dict:
        """Model üzerinde metrikleri hesapla."""
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["auc_roc"] = round(roc_auc_score(y_test, y_proba), 4)
        except Exception:
            metrics["auc_roc"] = None
        return metrics

    def train_one(self, algorithm: str, X, y) -> Optional[Dict]:
        """Tek bir algoritma için eğitim ve değerlendirme."""
        model = self._build_model(algorithm)
        if model is None:
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )

        # Logistic için ölçekleme
        if algorithm == "logistic_regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            metrics = self._evaluate(model, X_test_scaled, y_test)
            self.scalers[algorithm] = scaler
        else:
            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test)

        return {
            "model": model,
            "metrics": metrics,
            "algorithm": algorithm,
            "label": ALGORITHMS[algorithm]["label"],
        }

    def train_all(self, X, y, algorithms: Optional[list] = None) -> Dict:
        """Tüm algoritmaları eğit ve karşılaştır."""
        if algorithms is None:
            algorithms = list(ALGORITHMS.keys())

        results = {}
        for algo in algorithms:
            try:
                result = self.train_one(algo, X, y)
                if result is not None:
                    results[algo] = result
            except Exception as e:
                results[algo] = {"error": str(e), "algorithm": algo}

        return results

    def get_champion(self, results: Dict, metric: str = "f1") -> Optional[str]:
        """En iyi performans gösteren algoritmanın adını döndür."""
        scores = {
            algo: r["metrics"].get(metric, 0)
            for algo, r in results.items()
            if "metrics" in r and r["metrics"].get(metric) is not None
        }
        if not scores:
            return None
        return max(scores, key=scores.get)

    def evaluate_on_new_data(self, results: Dict, X_new, y_new) -> Dict:
        """Eğitilmiş tüm modelleri yeni veride test et."""
        evaluations = {}
        for algo, result in results.items():
            if "model" not in result:
                continue
            model = result["model"]
            try:
                if algo == "logistic_regression" and algo in self.scalers:
                    X_scaled = self.scalers[algo].transform(X_new)
                    metrics = self._evaluate(model, X_scaled, y_new)
                else:
                    metrics = self._evaluate(model, X_new, y_new)
                evaluations[algo] = metrics
            except Exception as e:
                evaluations[algo] = {"error": str(e)}
        return evaluations
