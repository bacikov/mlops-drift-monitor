"""
Streaming Drift Detectors for real-time change detection.

Implements online algorithms that process data one sample at a time:
- ADWIN (Adaptive Windowing)
- Page-Hinkley Test
- DDM (Drift Detection Method)
"""
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass, field


@dataclass
class StreamingDetectorState:
    """State of a streaming detector at any point in time."""
    detector_name: str
    is_drift: bool
    is_warning: bool
    statistic: float
    n_samples_seen: int
    details: Dict = field(default_factory=dict)


class ADWIN:
    """
    Adaptive Windowing (ADWIN) algorithm.
    
    Maintains a variable-length window of recent items and detects
    change by comparing the distributions of two sub-windows.
    When a change is detected, the older part is dropped.
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
        self._drift_detected = False
        self._n_detections = 0
    
    def update(self, value: float) -> bool:
        """
        Add a new observation and check for drift.
        Returns True if drift is detected.
        """
        self.window.append(value)
        self.total += value
        self.n += 1
        
        if self.n > 1:
            mean = self.total / self.n
            self.variance += (value - mean) ** 2
        
        self._drift_detected = False
        
        if self.n < 10:
            return False
        
        # Try different cut points
        n0 = 0
        sum0 = 0.0
        
        items_to_check = list(self.window)
        for i in range(len(items_to_check) - 1):
            n0 += 1
            sum0 += items_to_check[i]
            n1 = self.n - n0
            sum1 = self.total - sum0
            
            if n0 < 5 or n1 < 5:
                continue
            
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            # Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))
            
            if abs(mean0 - mean1) >= epsilon:
                # Drop the older sub-window
                for _ in range(n0):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.n -= 1
                
                self._drift_detected = True
                self._n_detections += 1
                self.variance = 0
                if self.n > 0:
                    mean = self.total / self.n
                    for v in self.window:
                        self.variance += (v - mean) ** 2
                break
        
        return self._drift_detected
    
    def get_state(self) -> StreamingDetectorState:
        return StreamingDetectorState(
            detector_name="ADWIN",
            is_drift=self._drift_detected,
            is_warning=False,
            statistic=self.total / max(self.n, 1),
            n_samples_seen=self.n,
            details={
                "window_size": len(self.window),
                "n_detections": self._n_detections,
                "delta": self.delta,
            },
        )
    
    def reset(self):
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
        self._drift_detected = False


class PageHinkley:
    """
    Page-Hinkley Test for change detection.
    
    Monitors cumulative difference between observed values 
    and their mean, triggering when it exceeds a threshold.
    """
    
    def __init__(self, threshold: float = 50.0, alpha: float = 0.005, min_samples: int = 30):
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
        
        self.n = 0
        self.sum = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative = float('inf')
        self._drift_detected = False
        self._n_detections = 0
    
    def update(self, value: float) -> bool:
        """Add observation and check for drift."""
        self.n += 1
        self.sum += value
        mean = self.sum / self.n
        
        self.cumulative_sum += value - mean - self.alpha
        self.min_cumulative = min(self.min_cumulative, self.cumulative_sum)
        
        self._drift_detected = False
        
        if self.n >= self.min_samples:
            ph_value = self.cumulative_sum - self.min_cumulative
            if ph_value > self.threshold:
                self._drift_detected = True
                self._n_detections += 1
                self.reset()
        
        return self._drift_detected
    
    def get_state(self) -> StreamingDetectorState:
        ph_value = self.cumulative_sum - self.min_cumulative if self.n > 0 else 0
        return StreamingDetectorState(
            detector_name="PageHinkley",
            is_drift=self._drift_detected,
            is_warning=ph_value > self.threshold * 0.7,
            statistic=ph_value,
            n_samples_seen=self.n,
            details={
                "threshold": self.threshold,
                "alpha": self.alpha,
                "n_detections": self._n_detections,
            },
        )
    
    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative = float('inf')


class DDM:
    """
    Drift Detection Method (DDM).
    
    Monitors the error rate of a model and detects drift when 
    the error rate significantly increases beyond historical minimum.
    """
    
    def __init__(self, min_samples: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level
        
        self.n = 0
        self.p = 0.0  # error rate
        self.s = 0.0  # standard deviation
        self.p_min = float('inf')
        self.s_min = float('inf')
        
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_errors = 0
    
    def update(self, is_error: bool) -> bool:
        """
        Add a new prediction result (0 = correct, 1 = error).
        Returns True if drift is detected.
        """
        self.n += 1
        if is_error:
            self._n_errors += 1
        
        self.p = self._n_errors / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n) if self.n > 0 else 0
        
        self._drift_detected = False
        self._warning_detected = False
        
        if self.n < self.min_samples:
            return False
        
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        
        current_level = self.p + self.s
        min_level = self.p_min + self.s_min
        
        if current_level > min_level + self.drift_level * self.s_min:
            self._drift_detected = True
            self._n_detections += 1
            self.reset()
        elif current_level > min_level + self.warning_level * self.s_min:
            self._warning_detected = True
        
        return self._drift_detected
    
    def get_state(self) -> StreamingDetectorState:
        return StreamingDetectorState(
            detector_name="DDM",
            is_drift=self._drift_detected,
            is_warning=self._warning_detected,
            statistic=self.p,
            n_samples_seen=self.n,
            details={
                "error_rate": round(self.p, 6),
                "std": round(self.s, 6),
                "min_error_rate": round(self.p_min, 6) if self.p_min < float('inf') else None,
                "n_detections": self._n_detections,
            },
        )
    
    def reset(self):
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self._n_errors = 0


class StreamingDriftManager:
    """
    Manages multiple streaming detectors across multiple features.
    Provides a unified interface for real-time drift monitoring.
    """
    
    def __init__(self, feature_names: List[str], adwin_delta=0.002, ph_threshold=50.0):
        self.feature_names = feature_names
        self.detectors: Dict[str, Dict] = {}
        
        for feat in feature_names:
            self.detectors[feat] = {
                "adwin": ADWIN(delta=adwin_delta),
                "page_hinkley": PageHinkley(threshold=ph_threshold),
            }
        
        # DDM works on model error, not per-feature
        self.ddm = DDM()
    
    def update_features(self, values: Dict[str, float]) -> Dict[str, List[StreamingDetectorState]]:
        """
        Update all detectors with new feature values.
        Returns drift states per feature.
        """
        results = {}
        for feat, val in values.items():
            if feat in self.detectors:
                states = []
                for name, detector in self.detectors[feat].items():
                    detector.update(val)
                    states.append(detector.get_state())
                results[feat] = states
        return results
    
    def update_prediction(self, is_error: bool) -> StreamingDetectorState:
        """Update DDM with a prediction result."""
        self.ddm.update(is_error)
        return self.ddm.get_state()
    
    def get_all_states(self) -> Dict:
        """Get current state of all detectors."""
        states = {}
        for feat in self.feature_names:
            states[feat] = {
                name: det.get_state() for name, det in self.detectors[feat].items()
            }
        states["_model_ddm"] = self.ddm.get_state()
        return states
    
    def any_drift_detected(self) -> bool:
        """Check if any detector has flagged drift."""
        for feat in self.feature_names:
            for det in self.detectors[feat].values():
                if det.get_state().is_drift:
                    return True
        if self.ddm.get_state().is_drift:
            return True
        return False
