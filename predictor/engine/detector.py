"""
Drift Detector
==============
İstatistiksel drift tespiti.

3 test paralel çalışır:
  1. KS Test (Kolmogorov-Smirnov)
     - İki dağılım arasındaki maksimum farkı ölçer
     - Büyük örneklemlerde p-value yerine istatistik kullanılır

  2. PSI (Population Stability Index)
     - Bankacılık sektörü standardı (Siddiqi 2006)
     - PSI < 0.10 → stabil, 0.10-0.20 → dikkat, > 0.20 → ciddi

  3. Wasserstein Distance (Earth Mover's Distance)
     - İki dağılımı birbirine taşımanın minimum maliyeti
     - Küçük kaymaları da yakalayan, robust bir metrik

Composite skor = (KS_norm + PSI_norm + Wasserstein_norm) / 3
Eşit ağırlık kullanılır — hangi test daha önemli sorusunun cevabı
veri setine göre değiştiğinden önyargısız yaklaşım tercih edilir.

Eşikler PSI endüstri standardına dayanarak belirlendi ve
airline + hava durumu veri setlerinde ampirik olarak doğrulandı.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats

from predictor.engine.types import DriftResult, FeatureDriftDetail
from predictor.config import DRIFT_CFG


class DriftDetector:
    """
    Feature dağılım değişikliklerini tespit eder.

    Her feature için 3 test çalışır.
    Composite skor eşit ağırlıkla hesaplanır.
    Severity PSI endüstri standardına göre belirlenir.
    """

    def __init__(self):
        self.cfg = DRIFT_CFG

    # ── Individual Tests ──────────────────────────────────────────

    def _ks_test(self, ref: np.ndarray, cur: np.ndarray):
        """
        Kolmogorov-Smirnov two-sample test.
        Returns (statistic, is_drifted).
        p-value < 0.05 → drifted.
        """
        stat, p = stats.ks_2samp(ref, cur)
        return float(stat), p < self.cfg.ks_p_value

    def _psi(self, ref: np.ndarray, cur: np.ndarray, n_bins: int = 50):
        """
        Population Stability Index.
        Returns (psi_value, is_drifted).
        PSI > 0.20 → significant shift (industry standard).
        """
        eps = 1e-10
        lo  = min(ref.min(), cur.min())
        hi  = max(ref.max(), cur.max())

        if lo == hi:
            return 0.0, False

        bins = np.linspace(lo, hi, n_bins + 1)
        r_cnt = np.histogram(ref, bins=bins)[0].astype(float) + eps
        c_cnt = np.histogram(cur, bins=bins)[0].astype(float) + eps
        r_pct = r_cnt / r_cnt.sum()
        c_pct = c_cnt / c_cnt.sum()

        psi = float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))
        return psi, psi > self.cfg.psi_critical

    def _wasserstein(self, ref: np.ndarray, cur: np.ndarray):
        """
        Wasserstein-1 distance (Earth Mover's Distance).
        Normalize edilmiş forma dönüştürülür (ref std'ye bölünür).
        Returns (normalized_distance, is_drifted).
        """
        dist = float(stats.wasserstein_distance(ref, cur))
        std  = float(np.std(ref))
        normalized = dist / std if std > 1e-10 else dist
        return normalized, normalized > self.cfg.wasserstein_threshold

    # ── Normalize ─────────────────────────────────────────────────

    def _normalize(self, test: str, value: float) -> float:
        """Test değerini 0-1 arasına normalize et."""
        upper = {
            "ks":          self.cfg.ks_norm_max,
            "psi":         self.cfg.psi_norm_max,
            "wasserstein": self.cfg.was_norm_max,
        }.get(test, 0.3)
        return min(1.0, value / upper)

    # ── Severity ──────────────────────────────────────────────────

    def _classify_severity(self, score: float) -> str:
        """
        Composite skoru severity'ye çevir.

        Eşikler PSI endüstri standardına dayanır:
          PSI < 0.10 → stabil (none/low)
          PSI 0.10-0.20 → dikkat (medium)
          PSI > 0.20 → ciddi kayma (high/critical)
        """
        if score >= self.cfg.severity_critical: return "critical"
        if score >= self.cfg.severity_high:     return "high"
        if score >= self.cfg.severity_medium:   return "medium"
        if score >= self.cfg.severity_low:      return "low"
        return "none"

    # ── Main ──────────────────────────────────────────────────────

    def detect(self,
               ref_df: pd.DataFrame,
               new_df: pd.DataFrame,
               feature_cols: List[str],
               ref_rate: Optional[float] = None,
               new_rate: Optional[float] = None) -> DriftResult:
        """
        Drift analizi yap.

        Args:
            ref_df:       Referans veri
            new_df:       Yeni veri
            feature_cols: Analiz edilecek feature'lar
            ref_rate:     Referanstaki gerçek pozitif oranı (%)
            new_rate:     Yeni verideki gerçek pozitif oranı (%)

        Returns:
            DriftResult
        """
        feature_scores   = {}
        feature_details  = []
        drifted_features = []

        for feat in feature_cols:
            if feat not in ref_df.columns or feat not in new_df.columns:
                continue

            ref_vals = ref_df[feat].dropna().values
            new_vals = new_df[feat].dropna().values

            if len(ref_vals) < 10 or len(new_vals) < 10:
                continue

            # 3 test
            ks_v,  ks_d  = self._ks_test(ref_vals, new_vals)
            psi_v, psi_d = self._psi(ref_vals, new_vals)
            wa_v,  wa_d  = self._wasserstein(ref_vals, new_vals)

            # Normalize
            norm_ks  = self._normalize("ks",  ks_v)
            norm_psi = self._normalize("psi", psi_v)
            norm_wa  = self._normalize("wasserstein", wa_v)

            # Composite (eşit ağırlık)
            score = (norm_ks + norm_psi + norm_wa) / 3
            feature_scores[feat] = round(score, 4)

            is_drifted = score > self.cfg.feature_drift_threshold
            if is_drifted:
                drifted_features.append(feat)

            feature_details.append(FeatureDriftDetail(
                feature=feat,
                ks_stat=round(ks_v, 5),  ks_drifted=ks_d,
                psi=round(psi_v, 5),     psi_drifted=psi_d,
                wasserstein=round(wa_v, 5), wa_drifted=wa_d,
                composite_score=round(score, 4),
                is_drifted=is_drifted,
            ))

        # Composite drift score
        composite = (
            float(np.mean(list(feature_scores.values())))
            if feature_scores else 0.0
        )

        # Rate drift
        rate_drift = None
        if ref_rate is not None and new_rate is not None:
            rate_drift = round(abs(new_rate - ref_rate), 2)

        return DriftResult(
            drift_score=round(composite, 4),
            severity=self._classify_severity(composite),
            drifted_features=drifted_features,
            feature_scores=feature_scores,
            feature_details=feature_details,
            rate_drift=rate_drift,
            n_features_total=len(feature_scores),
            n_features_drifted=len(drifted_features),
        )
