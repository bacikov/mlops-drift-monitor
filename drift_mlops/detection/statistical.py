"""
Statistical Drift Detection Module
===================================
3 istatistiksel test ile dağılım kayması tespiti:

1. KS Test (Kolmogorov-Smirnov)
   - En yaygın kullanılan iki-örneklem dağılım testi
   - İki dağılımın CDF'leri arasındaki maksimum farkı ölçer
   - H0: Her iki örnek aynı dağılımdan geliyor

2. PSI (Population Stability Index)
   - Bankacılık ve finans sektöründe 20+ yıldır standart metrik
   - Yorumlama: PSI < 0.10 → stabil, 0.10-0.20 → dikkat, > 0.20 → ciddi kayma
   - Referans: Yurdakul (2018), "Statistical Properties of PSI"

3. Wasserstein Distance (Earth Mover's Distance)
   - İki dağılımı birbirine dönüştürmek için gereken minimum "iş" miktarı
   - KS'e göre daha robust: küçük kaymaları da yakalıyor
   - Referans standardı normalize edilmiş forma dönüştürülüyor

NOT: KL Divergence, JS Divergence ve Chi-Square kaldırıldı.
- KL ve JS matematiksel olarak akraba (JS, KL'nin simetrik hali) — PSI benzer bilgiyi daha yorumlanabilir şekilde veriyor
- Chi-Square büyük örneklemlerde (n > 10.000) neredeyse her farkı anlamlı buluyor
  Bu klasik bir istatistiksel hatadır: effect size yerine p-value'ya odaklanmak.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DriftTestResult:
    """Tek bir test, tek bir feature için sonuç."""
    test_name: str
    feature: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    is_drifted: bool

    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "feature": self.feature,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
        }


class StatisticalDriftDetector:
    """
    3 istatistiksel test ile dağılım kayması tespiti.
    Her test bağımsız bir sinyal sağlar, composite skor eşit ağırlıkla birleştirilir.
    """

    def __init__(self, n_bins: int = 50, epsilon: float = 1e-10):
        self.n_bins = n_bins
        self.epsilon = epsilon

    # ── KS Test ───────────────────────────────────────────────────────────────
    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        p_threshold: float = 0.05,
    ) -> DriftTestResult:
        """
        İki-örneklem KS testi.
        p-value < eşik → dağılımlar anlamlı düzeyde farklı.

        Büyük örneklemlerde (n > 10.000) p-value çok küçük çıkar —
        bu nedenle sadece p-value değil, istatistik (D) değeri de raporlanır.
        Gerçek büyüklük için Wasserstein daha güvenilirdir.
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return DriftTestResult(
            test_name="ks_test",
            feature="",
            statistic=statistic,
            p_value=p_value,
            threshold=p_threshold,
            is_drifted=p_value < p_threshold,
        )

    # ── PSI ───────────────────────────────────────────────────────────────────
    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.2,
        n_bins: Optional[int] = None,
    ) -> DriftTestResult:
        """
        Population Stability Index.
        Endüstri standardı yorumlama:
          PSI < 0.10 → dağılım stabil
          PSI 0.10-0.20 → dikkat edilmeli
          PSI > 0.20 → ciddi kayma, müdahale gerekli

        Kaynak: Siddiqi (2006), "Credit Risk Scorecards"
        """
        bins = n_bins or self.n_bins
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            bins + 1,
        )

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = (ref_counts + self.epsilon) / (ref_counts.sum() + self.epsilon * bins)
        cur_pct = (cur_counts + self.epsilon) / (cur_counts.sum() + self.epsilon * bins)

        psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        return DriftTestResult(
            test_name="psi",
            feature="",
            statistic=psi_value,
            p_value=None,
            threshold=threshold,
            is_drifted=psi_value > threshold,
        )

    # ── Wasserstein Distance ───────────────────────────────────────────────────
    def wasserstein(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.1,
        normalize: bool = True,
    ) -> DriftTestResult:
        """
        Wasserstein-1 distance (Earth Mover's Distance).
        İki dağılımı birbirine "taşımanın" minimum maliyeti.

        Normalize=True: referansın standart sapmasına bölerek
        farklı ölçekli feature'ları karşılaştırılabilir hale getirir.
        """
        distance = float(stats.wasserstein_distance(reference, current))

        if normalize and np.std(reference) > 0:
            distance = distance / np.std(reference)

        return DriftTestResult(
            test_name="wasserstein",
            feature="",
            statistic=distance,
            p_value=None,
            threshold=threshold,
            is_drifted=distance > threshold,
        )

    # ── Tüm testleri çalıştır ─────────────────────────────────────────────────
    def run_all_tests(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "",
        thresholds: Optional[Dict] = None,
    ) -> List[DriftTestResult]:
        """3 testi çalıştır, feature ismini ata."""
        th = thresholds or {}

        results = [
            self.ks_test(reference, current, th.get("ks_p_value", 0.05)),
            self.psi(reference, current, th.get("psi_threshold", 0.2)),
            self.wasserstein(reference, current, th.get("wasserstein_threshold", 0.1)),
        ]

        for r in results:
            r.feature = feature_name

        return results

    # ── DataFrame üzerinde çalıştır ───────────────────────────────────────────
    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        thresholds: Optional[Dict] = None,
    ) -> Dict[str, List[DriftTestResult]]:
        """
        Tüm feature'lar üzerinde 3 testi çalıştır.
        Döndürür: {feature_name: [DriftTestResult, ...]}
        """
        results = {}
        common_features = [c for c in reference_df.columns if c in current_df.columns]

        for feature in common_features:
            ref_vals = reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            results[feature] = self.run_all_tests(
                ref_vals, cur_vals,
                feature_name=feature,
                thresholds=thresholds,
            )

        return results