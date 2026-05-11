"""
Mitigation Engine: Decides and executes actions based on drift severity.
Handles auto-retraining, feature updates, and fallback model switching.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.base import BaseEstimator, clone

from drift_mlops.config.settings import (
    DriftSeverity, MitigationAction, SEVERITY_ACTION_MAP,
    MODEL_CONFIG, PIPELINE_CONFIG
)
from drift_mlops.detection.scorer import DriftReport


@dataclass
class MitigationRecord:
    """Record of a mitigation action taken."""
    timestamp: datetime
    severity: DriftSeverity
    actions_taken: List[MitigationAction]
    trigger_score: float
    drifted_features: List[str]
    result: str  # "success", "failed", "skipped"
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "actions": [a.value for a in self.actions_taken],
            "trigger_score": round(self.trigger_score, 4),
            "drifted_features": self.drifted_features,
            "result": self.result,
            "details": self.details,
        }


class MitigationEngine:
    """
    Decision engine that maps drift reports to mitigation actions.
    Manages cooldowns, action history, and execution.
    """
    
    def __init__(
        self,
        severity_map: Optional[Dict] = None,
        cooldown_seconds: int = 300,
    ):
        self.severity_map = severity_map or SEVERITY_ACTION_MAP.mapping
        self.cooldown_seconds = cooldown_seconds
        self.history: List[MitigationRecord] = []
        self._last_action_time: Dict[MitigationAction, float] = {}
        self._action_handlers = {}
    
    def register_handler(self, action: MitigationAction, handler):
        """Register a callback function for a specific action."""
        self._action_handlers[action] = handler
    
    def evaluate(self, drift_report: DriftReport) -> List[MitigationAction]:
        """
        Given a drift report, determine which actions to take.
        Respects cooldown periods.
        """
        severity = drift_report.severity
        candidate_actions = self.severity_map.get(severity, [MitigationAction.NO_ACTION])
        
        # Filter by cooldown
        now = time.time()
        executable_actions = []
        for action in candidate_actions:
            last_time = self._last_action_time.get(action, 0)
            if now - last_time >= self.cooldown_seconds:
                executable_actions.append(action)
        
        return executable_actions
    
    def execute(
        self,
        drift_report: DriftReport,
        model: Optional[BaseEstimator] = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_new: Optional[pd.DataFrame] = None,
        y_new: Optional[pd.Series] = None,
    ) -> MitigationRecord:
        """
        Evaluate drift report and execute appropriate mitigation actions.
        """
        actions = self.evaluate(drift_report)
        
        if not actions or actions == [MitigationAction.NO_ACTION]:
            record = MitigationRecord(
                timestamp=datetime.now(),
                severity=drift_report.severity,
                actions_taken=[MitigationAction.NO_ACTION],
                trigger_score=drift_report.composite_score,
                drifted_features=drift_report.drifted_features,
                result="skipped",
            )
            self.history.append(record)
            return record
        
        details = {}
        result = "success"
        now = time.time()
        
        for action in actions:
            try:
                if action in self._action_handlers:
                    handler_result = self._action_handlers[action](
                        drift_report=drift_report,
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_new=X_new,
                        y_new=y_new,
                    )
                    details[action.value] = handler_result
                
                self._last_action_time[action] = now
                
            except Exception as e:
                details[action.value] = {"error": str(e)}
                result = "failed"
        
        record = MitigationRecord(
            timestamp=datetime.now(),
            severity=drift_report.severity,
            actions_taken=actions,
            trigger_score=drift_report.composite_score,
            drifted_features=drift_report.drifted_features,
            result=result,
            details=details,
        )
        
        self.history.append(record)
        return record
    
    def get_action_summary(self) -> Dict:
        """Summarize all actions taken."""
        if not self.history:
            return {"total_actions": 0}
        
        action_counts = {}
        for record in self.history:
            for action in record.actions_taken:
                action_counts[action.value] = action_counts.get(action.value, 0) + 1
        
        return {
            "total_records": len(self.history),
            "action_counts": action_counts,
            "last_action": self.history[-1].to_dict() if self.history else None,
            "success_rate": sum(1 for r in self.history if r.result == "success") / len(self.history),
        }


class ModelRetrainer:
    """
    Handles model retraining logic: full retrain, incremental update,
    and weighted sampling based on recency.
    """
    
    def __init__(self, base_model: BaseEstimator):
        self.base_model = base_model
        self.model_versions: List[Dict] = []
        self._current_version = 0
    
    def full_retrain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Tuple[BaseEstimator, Dict]:
        """Full retrain on combined old + new data."""
        new_model = clone(self.base_model)
        new_model.fit(X, y)
        
        self._current_version += 1
        version_info = {
            "version": self._current_version,
            "type": "full_retrain",
            "n_samples": len(X),
            "timestamp": datetime.now().isoformat(),
        }
        self.model_versions.append(version_info)
        
        return new_model, version_info
    
    def incremental_retrain(
        self,
        model: BaseEstimator,
        X_old: pd.DataFrame,
        y_old: pd.Series,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        new_weight: float = 0.7,
    ) -> Tuple[BaseEstimator, Dict]:
        """
        Retrain with weighted sampling favoring recent data.
        """
        # Create weighted combined dataset
        n_old = int(len(X_old) * (1 - new_weight))
        n_new = len(X_new)
        
        if n_old > 0:
            old_indices = np.random.choice(len(X_old), size=n_old, replace=False)
            X_combined = pd.concat([X_old.iloc[old_indices], X_new], ignore_index=True)
            y_combined = pd.concat([y_old.iloc[old_indices], y_new], ignore_index=True)
        else:
            X_combined = X_new
            y_combined = y_new
        
        new_model = clone(self.base_model)
        new_model.fit(X_combined, y_combined)
        
        self._current_version += 1
        version_info = {
            "version": self._current_version,
            "type": "incremental_retrain",
            "n_old_samples": n_old,
            "n_new_samples": n_new,
            "new_weight": new_weight,
            "timestamp": datetime.now().isoformat(),
        }
        self.model_versions.append(version_info)
        
        return new_model, version_info
    
    def get_version_history(self) -> List[Dict]:
        return self.model_versions


class FallbackManager:
    """
    Manages champion/challenger model pairs and fallback switching.
    """
    
    def __init__(self):
        self.champion: Optional[BaseEstimator] = None
        self.challenger: Optional[BaseEstimator] = None
        self.champion_score: float = 0.0
        self.challenger_score: float = 0.0
        self._is_fallback_active = False
    
    def set_champion(self, model: BaseEstimator, score: float):
        self.champion = model
        self.champion_score = score
    
    def set_challenger(self, model: BaseEstimator, score: float):
        self.challenger = model
        self.challenger_score = score
    
    def should_switch(self, champion_current_score: float) -> bool:
        """Check if we should switch to challenger."""
        if self.challenger is None:
            return False
        
        # Switch if champion dropped significantly and challenger is better
        drop = self.champion_score - champion_current_score
        if drop > MODEL_CONFIG.performance_drop_threshold:
            if self.challenger_score > champion_current_score:
                return True
        return False
    
    def switch_to_challenger(self) -> Optional[BaseEstimator]:
        """Promote challenger to champion."""
        if self.challenger is None:
            return None
        
        self.champion = self.challenger
        self.champion_score = self.challenger_score
        self.challenger = None
        self.challenger_score = 0.0
        self._is_fallback_active = True
        
        return self.champion
    
    def get_active_model(self) -> BaseEstimator:
        return self.champion
    
    @property
    def is_fallback_active(self) -> bool:
        return self._is_fallback_active
