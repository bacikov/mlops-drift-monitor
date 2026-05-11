"""
Feature Store: Manages reference distributions and live data windows.
Stores statistical summaries for comparison during drift detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import json
import os


@dataclass
class FeatureStats:
    """Statistical summary for a single feature."""
    name: str
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    histogram: np.ndarray = None
    bin_edges: np.ndarray = None
    n_samples: int = 0


class FeatureStore:
    """
    Stores and manages reference distributions and live data windows.
    Provides statistical baselines for drift comparison.
    """
    
    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, FeatureStats] = {}
        self.reference_target: Optional[pd.Series] = None
        self.live_buffer: deque = deque(maxlen=10000)
        self._version = 0
    
    def set_reference(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Store reference dataset and compute baseline statistics."""
        self.reference_data = X.copy()
        self.reference_target = y.copy() if y is not None else None
        self.reference_stats = {}
        
        for col in X.columns:
            values = X[col].dropna().values
            hist, bin_edges = np.histogram(values, bins=self.n_bins, density=True)
            
            self.reference_stats[col] = FeatureStats(
                name=col,
                mean=np.mean(values),
                std=np.std(values),
                median=np.median(values),
                min_val=np.min(values),
                max_val=np.max(values),
                q25=np.percentile(values, 25),
                q75=np.percentile(values, 75),
                histogram=hist,
                bin_edges=bin_edges,
                n_samples=len(values),
            )
        
        self._version += 1
    
    def add_live_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Add incoming data to the live buffer."""
        for idx in range(len(X)):
            record = {"X": X.iloc[idx].to_dict()}
            if y is not None:
                record["y"] = y.iloc[idx]
            self.live_buffer.append(record)
    
    def get_live_window(self, window_size: int = 500) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Get the most recent live data window."""
        recent = list(self.live_buffer)[-window_size:]
        if not recent:
            return pd.DataFrame(), None
        
        X = pd.DataFrame([r["X"] for r in recent])
        y = None
        if "y" in recent[0]:
            y = pd.Series([r["y"] for r in recent], name="target")
        
        return X, y
    
    def get_reference_distribution(self, feature: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get the reference histogram for a feature."""
        stats = self.reference_stats.get(feature)
        if stats is None:
            raise ValueError(f"No reference stats for feature: {feature}")
        return stats.histogram, stats.bin_edges
    
    def compute_live_distribution(
        self, feature: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram for live data using reference bin edges."""
        ref_stats = self.reference_stats.get(feature)
        if ref_stats is None:
            raise ValueError(f"No reference stats for feature: {feature}")
        
        values = data[feature].dropna().values
        hist, _ = np.histogram(values, bins=ref_stats.bin_edges, density=True)
        return hist, ref_stats.bin_edges
    
    def update_reference(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Update reference with new data (after successful retrain)."""
        self.set_reference(X, y)
    
    @property
    def version(self) -> int:
        return self._version
    
    @property
    def feature_names(self) -> list:
        if self.reference_data is not None:
            return list(self.reference_data.columns)
        return []
    
    def get_summary(self) -> Dict:
        """Get a summary of the feature store state."""
        return {
            "version": self._version,
            "n_features": len(self.reference_stats),
            "reference_samples": self.reference_stats[self.feature_names[0]].n_samples if self.feature_names else 0,
            "live_buffer_size": len(self.live_buffer),
            "features": self.feature_names,
        }
