"""
Synthetic Data Generator with Configurable Drift Injection.

Generates a base dataset and progressively injects different types of drift:
- Covariate drift (input distribution shift)
- Concept drift (P(y|x) changes)
- Gradual vs sudden drift
- Feature-level drift
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):
    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class DriftConfig:
    """Configuration for drift injection."""
    drift_type: DriftType = DriftType.NONE
    drift_magnitude: float = 0.0        # 0.0 to 1.0
    affected_features: List[int] = None  # indices of features to drift
    drift_start_ratio: float = 0.5       # when drift starts (as fraction of data)
    concept_drift: bool = False          # whether to change P(y|x)


class DataGenerator:
    """
    Generates synthetic classification data with controllable drift injection.
    Uses a credit-scoring-like scenario for interpretability.
    """
    
    FEATURE_NAMES = [
        "income", "age", "credit_score", "debt_ratio",
        "employment_years", "num_accounts", "loan_amount",
        "interest_rate", "monthly_payment", "savings_balance"
    ]
    
    def __init__(self, n_features: int = 10, random_state: int = 42):
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
        self._base_means = self.rng.uniform(0, 10, n_features)
        self._base_stds = self.rng.uniform(0.5, 3, n_features)
        # Scale weights down so logits stay in a reasonable range for sigmoid
        self._base_weights = self.rng.randn(n_features) * 0.1
        self._base_bias = 0.0
    
    def generate_reference(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate the reference (baseline) dataset."""
        X = np.column_stack([
            self.rng.normal(self._base_means[i], self._base_stds[i], n_samples)
            for i in range(self.n_features)
        ])
        y = self._compute_labels(X, self._base_weights, self._base_bias)
        
        df = pd.DataFrame(X, columns=self.FEATURE_NAMES[:self.n_features])
        return df, pd.Series(y, name="target")
    
    def generate_stream(
        self,
        n_samples: int = 1000,
        drift_config: Optional[DriftConfig] = None
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Generate streaming data with optional drift.
        
        Returns:
            X: Feature DataFrame
            y: Target Series
            drift_labels: Array indicating drift status per sample (0=no drift, 1=drift)
        """
        if drift_config is None:
            drift_config = DriftConfig()
        
        X = np.column_stack([
            self.rng.normal(self._base_means[i], self._base_stds[i], n_samples)
            for i in range(self.n_features)
        ])
        
        drift_labels = np.zeros(n_samples, dtype=int)
        weights = self._base_weights.copy()
        
        if drift_config.drift_type != DriftType.NONE:
            affected = drift_config.affected_features or list(range(min(3, self.n_features)))
            start_idx = int(n_samples * drift_config.drift_start_ratio)
            magnitude = drift_config.drift_magnitude
            
            if drift_config.drift_type == DriftType.SUDDEN:
                # Abrupt shift in feature distributions
                for feat_idx in affected:
                    shift = magnitude * self._base_stds[feat_idx] * 3
                    X[start_idx:, feat_idx] += shift
                drift_labels[start_idx:] = 1
                
            elif drift_config.drift_type == DriftType.GRADUAL:
                # Progressive shift over remaining samples
                n_drift = n_samples - start_idx
                for feat_idx in affected:
                    shift = magnitude * self._base_stds[feat_idx] * 3
                    gradual_shift = np.linspace(0, shift, n_drift)
                    X[start_idx:, feat_idx] += gradual_shift
                # Mark as drift once shift exceeds 30% of max
                drift_threshold = int(start_idx + n_drift * 0.3)
                drift_labels[drift_threshold:] = 1
                
            elif drift_config.drift_type == DriftType.INCREMENTAL:
                # Small stepwise changes
                n_steps = 5
                step_size = (n_samples - start_idx) // n_steps
                for step in range(n_steps):
                    step_start = start_idx + step * step_size
                    step_end = step_start + step_size
                    for feat_idx in affected:
                        shift = magnitude * self._base_stds[feat_idx] * (step + 1) / n_steps
                        X[step_start:step_end, feat_idx] += shift
                drift_labels[start_idx:] = 1
                
            elif drift_config.drift_type == DriftType.RECURRING:
                # Drift that comes and goes
                period = (n_samples - start_idx) // 4
                for cycle in range(4):
                    cycle_start = start_idx + cycle * period
                    cycle_end = cycle_start + period
                    if cycle % 2 == 0:  # drift on
                        for feat_idx in affected:
                            shift = magnitude * self._base_stds[feat_idx] * 3
                            X[cycle_start:cycle_end, feat_idx] += shift
                        drift_labels[cycle_start:cycle_end] = 1
            
            # Concept drift: change the decision boundary
            if drift_config.concept_drift:
                weights = self._base_weights.copy()
                for feat_idx in affected:
                    weights[feat_idx] *= -1  # flip feature importance
        
        y = self._compute_labels(X, weights, self._base_bias)
        
        df = pd.DataFrame(X, columns=self.FEATURE_NAMES[:self.n_features])
        return df, pd.Series(y, name="target"), drift_labels
    
    def _compute_labels(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        bias: float
    ) -> np.ndarray:
        """Compute binary labels using logistic function with centered logits."""
        logits = X @ weights + bias
        # Center logits around 0 so we get ~50/50 class balance
        logits = logits - np.mean(logits)
        probs = 1 / (1 + np.exp(-logits))
        noise = self.rng.uniform(0, 1, len(probs))
        return (probs > noise).astype(int)
    
    def generate_batch_stream(
        self,
        n_batches: int = 20,
        batch_size: int = 100,
        drift_config: Optional[DriftConfig] = None
    ) -> List[Dict]:
        """
        Generate data as a list of batches (simulating real-time stream).
        Each batch is a dict with 'X', 'y', 'drift_label', 'batch_id', 'timestamp'.
        """
        total = n_batches * batch_size
        X, y, drift_labels = self.generate_stream(total, drift_config)
        
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batches.append({
                "batch_id": i,
                "timestamp": pd.Timestamp.now() + pd.Timedelta(seconds=i * 5),
                "X": X.iloc[start:end].reset_index(drop=True),
                "y": y.iloc[start:end].reset_index(drop=True),
                "drift_label": drift_labels[start:end],
            })
        
        return batches
