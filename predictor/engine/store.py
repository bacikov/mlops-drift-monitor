"""
Store
=====
Model versiyonlama ve geçmiş kayıt.

ModelStore:
  - Her eğitim/retrain'i diske kaydeder
  - Her algoritma ayrı .pkl dosyası
  - Meta bilgi JSON olarak saklanır
  - Versiyonlar timestamp ile adlandırılır

HistoryStore:
  - Her analiz sonucunu JSON'a ekler
  - Session kapansa da kalır
  - Grafik ve tablo için okunur
"""

import os, json, pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from predictor.engine.types import TrainResult, AnalysisRecord
from predictor.config import ALGORITHMS


class ModelStore:
    """Model versiyonlarını diske yönetir."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self,
             train_results: Dict[str, TrainResult],
             champion: str,
             label: str = "") -> str:
        """
        Tüm modelleri kaydet.

        Her algoritma için ayrı .pkl.
        meta.json içinde metrics, feature_cols, vs.

        Returns:
            version string (timestamp)
        """
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        vdir = os.path.join(self.base_dir, ts)
        os.makedirs(vdir, exist_ok=True)

        saved = []
        for algo, r in train_results.items():
            path = os.path.join(vdir, f"{algo}.pkl")
            with open(path, "wb") as f:
                pickle.dump(r, f)
            saved.append(algo)

        meta = {
            "version":      ts,
            "champion":     champion,
            "algorithms":   saved,
            "label":        label,
            "saved_at":     datetime.now().isoformat(),
            "feature_cols": list(train_results[champion].feature_cols) if champion in train_results else [],
            "target_col":   train_results[champion].target_col if champion in train_results else "",
            "metrics": {
                a: r.metrics for a, r in train_results.items()
            },
            "rates": {
                a: {
                    "train_rate":     r.train_rate,
                    "predicted_rate": r.predicted_rate,
                }
                for a, r in train_results.items()
            },
        }
        with open(os.path.join(vdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return ts

    def list_versions(self) -> List[Dict]:
        """Kaydedilmiş versiyonları listele (yeniden eskiye)."""
        versions = []
        if not os.path.exists(self.base_dir):
            return versions
        for d in sorted(os.listdir(self.base_dir), reverse=True):
            mp = os.path.join(self.base_dir, d, "meta.json")
            if os.path.exists(mp):
                try:
                    with open(mp) as f:
                        versions.append(json.load(f))
                except Exception:
                    pass
        return versions

    def load(self, version: str) -> Tuple[Dict[str, TrainResult], Dict]:
        """Belirli bir versiyonu yükle."""
        vdir = os.path.join(self.base_dir, version)
        with open(os.path.join(vdir, "meta.json")) as f:
            meta = json.load(f)

        results = {}
        for algo in meta.get("algorithms", []):
            path = os.path.join(vdir, f"{algo}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    results[algo] = pickle.load(f)

        return results, meta

    def get_pkl_bytes(self, version: str, algo: str) -> Optional[bytes]:
        """Belirli bir algoritmanın pkl bytes'ını döndür (download için)."""
        path = os.path.join(self.base_dir, version, f"{algo}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    def delete_version(self, version: str):
        """Bir versiyonu sil."""
        import shutil
        vdir = os.path.join(self.base_dir, version)
        if os.path.exists(vdir):
            shutil.rmtree(vdir)


class HistoryStore:
    """Analiz geçmişini diske yazar ve okur."""

    def __init__(self, path: str):
        self.path = path

    def append(self, record: AnalysisRecord):
        """Yeni bir analiz kaydı ekle."""
        history = self.load()
        history.append({
            "timestamp":        record.timestamp,
            "dataset_name":     record.dataset_name,
            "use_case":         record.use_case,
            "champion":         record.champion,
            "drift_score":      record.drift_score,
            "severity":         record.severity,
            "ref_rate":         record.ref_rate,
            "predicted_rate":   record.predicted_rate,
            "actual_rate":      record.actual_rate,
            "rate_error":       record.rate_error,
            "f1":               record.f1,
            "auc_roc":          record.auc_roc,
            "drifted_features": record.drifted_features,
            "retrained":        record.retrained,
            "model_version":    record.model_version,
            "all_algo_rates":   record.all_algo_rates,
            "all_algo_f1s":     record.all_algo_f1s,
        })
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def load(self) -> List[Dict]:
        """Tüm geçmişi yükle."""
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return []

    def clear(self):
        """Tüm geçmişi sil."""
        if os.path.exists(self.path):
            os.remove(self.path)
