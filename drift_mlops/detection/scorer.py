"""
Drift Scorer
============
3 testin sonuçlarını tek bir composite skora dönüştürür.

Composite Skor Hesaplama:
  Her test için normalize skor (0-1 arası) hesaplanır.
  Composite = ortalama(KS_norm, PSI_norm, Wasserstein_norm)
  Eşit ağırlık kullanılır çünkü "hangi test daha önemli?" sorusunun
  cevabı veri setine göre değişir — önyargısız yaklaşım eşit ağırlık.

Eşik Değerleri:
  PSI'ın endüstri standardı yorumlama kılavuzu referans alındı:
    PSI < 0.10 → stabil
    PSI 0.10-0.20 → dikkat
    PSI > 0.20 → ciddi kayma
  Bu mantık composite skora uyarlandı ve hava durumu veri seti
  üzerinde ampirik olarak doğrulandı.

  none:     composite < 0.15
  low:      0.15 - 0.30
  medium:   0.30 - 0.50
  high:     0.50 - 0.70
  critical: > 0.70
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from drift_mlops.detection.statistical import DriftTestResult
from drift_mlops.config.settings import DriftSeverity, DRIFT_THRESHOLDS


@dataclass
class DriftReport:
    """Tek bir analiz penceresi için tam drift raporu."""
    timestamp: datetime
    composite_score: float
    severity: DriftSeverity
    feature_scores: Dict[str, float]
    drifted_features: List[str]
    n_tests_total: int
    n_tests_drifted: int
    statistical_results: Dict[str, List[Dict]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "composite_score": round(self.composite_score, 4),
            "severity": self.severity.value,
            "drifted_features": self.drifted_features,
            "feature_scores": {k: round(v, 4) for k, v in self.feature_scores.items()},
            "n_tests_total": self.n_tests_total,
            "n_tests_drifted": self.n_tests_drifted,
        }


class DriftScorer:
    """
    KS, PSI, Wasserstein sonuçlarını composite skora dönüştürür.

    Her test normalize edilerek 0-1 arasına çekilir:
      - KS: istatistik doğrudan 0-1 arası (max fark)
      - PSI: eşiğe (0.2) oranlanır, 1'de doyurulur
      - Wasserstein: eşiğe (0.1) oranlanır, 1'de doyurulur

    Composite = (KS_norm + PSI_norm + Wasserstein_norm) / 3
    """

    # Normalleştirme için referans eşikler
    # Bu değerlerin üstü = tam drift (skor = 1.0)
    NORM_THRESHOLDS = {
        "ks_test": 0.3,        # KS istatistiği 0.3+ → tam drift
        "psi": 0.4,            # PSI 0.4+ → tam drift (0.2 = ciddi, 0.4 = çok ciddi)
        "wasserstein": 0.3,    # Normalize Wasserstein 0.3+ → tam drift
    }

    def __init__(self):
        self.history: List[DriftReport] = []

    def _normalize(self, result: DriftTestResult) -> float:
        """
        Test sonucunu 0-1 arasına normalize et.
        KS için p-value yerine istatistik kullanılır —
        büyük örneklemlerde p-value her zaman < 0.05 olur, yanıltıcı.
        """
        norm_threshold = self.NORM_THRESHOLDS.get(result.test_name, result.threshold)

        if result.test_name == "ks_test":
            # KS statistic zaten 0-1 arası
            return min(1.0, result.statistic / norm_threshold)
        else:
            # PSI ve Wasserstein: eşiğe oranla
            return min(1.0, result.statistic / max(norm_threshold, 1e-10))

    def score(
        self,
        statistical_results: Dict[str, List[DriftTestResult]],
        streaming_results=None,  # geriye dönük uyumluluk için tutuldu
    ) -> DriftReport:
        """
        Composite drift skoru hesapla.

        Args:
            statistical_results: {feature: [DriftTestResult, ...]}

        Returns:
            DriftReport
        """
        feature_scores = {}
        drifted_features = []
        total_tests = 0
        drifted_tests = 0

        for feature, test_results in statistical_results.items():
            if not test_results:
                continue

            # Her testin normalize skorunu al, eşit ağırlıkla ortala
            norm_scores = [self._normalize(r) for r in test_results]
            feature_score = float(np.mean(norm_scores))
            feature_scores[feature] = feature_score

            for r in test_results:
                total_tests += 1
                if r.is_drifted:
                    drifted_tests += 1

            # Feature drifted sayılır eğer ortalaması 0.5'in üstündeyse
            if feature_score > 0.5:
                drifted_features.append(feature)

        # Composite skor = tüm feature skorlarının ortalaması
        composite = float(np.mean(list(feature_scores.values()))) if feature_scores else 0.0

        severity = self._classify_severity(composite)

        report = DriftReport(
            timestamp=datetime.now(),
            composite_score=composite,
            severity=severity,
            feature_scores=feature_scores,
            drifted_features=drifted_features,
            n_tests_total=total_tests,
            n_tests_drifted=drifted_tests,
            statistical_results={
                feat: [r.to_dict() for r in results]
                for feat, results in statistical_results.items()
            },
        )

        self.history.append(report)
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        return report

    def _classify_severity(self, score: float) -> DriftSeverity:
        """
        Composite skoru severity seviyesine dönüştür.

        Eşikler PSI endüstri standardına dayanır ve
        hava durumu veri seti üzerinde ampirik olarak doğrulanmıştır.
        Farklı domainlerde yeniden kalibre edilmesi önerilir.
        """
        if score >= 0.70:
            return DriftSeverity.CRITICAL
        elif score >= 0.50:
            return DriftSeverity.HIGH
        elif score >= 0.30:
            return DriftSeverity.MEDIUM
        elif score >= 0.15:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def get_trend(self, n_last: int = 10) -> Dict:
        """Son N raporun drift skor trendini döndür."""
        recent = self.history[-n_last:]
        if not recent:
            return {"scores": [], "severities": [], "trend": "stable"}

        scores = [r.composite_score for r in recent]
        trend = "stable"
        if len(scores) >= 3:
            if scores[-1] > scores[0] * 1.5:
                trend = "increasing"
            elif scores[-1] < scores[0] * 0.5:
                trend = "decreasing"

        return {
            "scores": [round(s, 4) for s in scores],
            "severities": [r.severity.value for r in recent],
            "trend": trend,
            "latest": round(scores[-1], 4) if scores else 0,
        }