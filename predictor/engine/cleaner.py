"""
Data Cleaner
============
CSV verisini modele hazır hale getirir.

- Sayısal sütunları seçer
- Target sütununu binary yapar
- Eksik değerleri doldurur
- Feature / target ayırır
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class DataCleaner:
    """
    Ham CSV'yi temizler ve modele hazır hale getirir.

    Binary hedef oluşturma kuralı:
      - Zaten binary (0/1 veya iki unique değer) → küçük olanı 0, büyük olanı 1
      - Sürekli değer → medyanın üstü 1, altı 0
    """

    @staticmethod
    def clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Ham DataFrame'i temizle."""
        # Sayısal sütunları al
        df_c = df.select_dtypes(include=[np.number]).copy()

        # Target sayısal değilse ekle
        if target_col not in df_c.columns and target_col in df.columns:
            df_c[target_col] = df[target_col]

        # Eksik değerleri medyanla doldur (target hariç)
        for col in df_c.columns:
            if col != target_col and df_c[col].isnull().any():
                df_c[col] = df_c[col].fillna(df_c[col].median())

        # Target eksikse o satırları kaldır
        df_c = df_c.dropna(subset=[target_col])

        # Task otomatik belirle
        vals   = df_c[target_col]
        unique = sorted(vals.dropna().unique())
        n_unique = len(unique)

        # Regression mi classification mi?
        # Unique değer 2'den fazla VE max değer 100'den büyükse regression
        if n_unique > 2 and vals.max() > 100:
            # Regression — olduğu gibi bırak
            df_c['_task'] = 'regression'
        else:
            # Classification — binary yap
            if n_unique == 0:
                raise ValueError(f"Target column '{target_col}' has no valid values.")
            elif n_unique == 1:
                raise ValueError(f"Target column '{target_col}' has only one unique value: {unique[0]}")
            elif n_unique == 2:
                df_c[target_col] = (vals > unique[0]).astype(int)
            else:
                df_c[target_col] = (vals > vals.median()).astype(int)
            df_c['_task'] = 'classification'

        return df_c

    @staticmethod
    def get_feature_cols(df: pd.DataFrame, target_col: str) -> List[str]:
        """Target dışındaki tüm sütunları döndür."""
        return [c for c in df.columns if c != target_col]

    @staticmethod
    def prepare_features(df: pd.DataFrame,
                         target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """X ve y'yi ayır."""
        feature_cols = DataCleaner.get_feature_cols(df, target_col)
        return df[feature_cols], df[target_col]

    @staticmethod
    def align_features(new_df: pd.DataFrame,
                       feature_cols: List[str]) -> pd.DataFrame:
        """
        Yeni veriyi eğitimdeki feature sütunlarına hizala.
        Eksik sütunlar 0 ile doldurulur.
        """
        missing = [c for c in feature_cols if c not in new_df.columns]
        if missing:
            raise ValueError(
                f"New data is missing {len(missing)} feature(s): {missing[:5]}..."
                f" Make sure you upload data with the same columns as training data."
            )
        return new_df[feature_cols].copy()

    @staticmethod
    def get_positive_rate(series: pd.Series) -> float:
        """Bir serinin pozitif oranını (%) döndür."""
        return round(float(series.mean()) * 100, 2)