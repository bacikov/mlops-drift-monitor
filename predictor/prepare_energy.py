"""
PJME Energy Consumption - Veri Hazırlama
=========================================
PJM East bölgesi saatlik enerji tüketimi (2002-2018).

Feature'lar (hepsi tahmin anında biliniyor):
  hour          — saat (0-23)
  dayofweek     — haftanın günü (0=Pazartesi, 6=Pazar)
  month         — ay (1-12)
  quarter       — çeyrek (1-4)
  year          — yıl
  dayofyear     — yılın günü (1-365)
  weekofyear    — yılın haftası
  is_weekend    — hafta sonu mu?
  is_night      — gece mi? (22:00-06:00)
  is_morning    — sabah mı? (06:00-10:00)
  is_peak       — yoğun saat mi? (07:00-21:00)
  season        — mevsim (0=kış, 1=ilkbahar, 2=yaz, 3=sonbahar)

Target: PJME_MW (megawatt cinsinden enerji tüketimi)

Drift senaryosu:
  Referans: 2002-2015 (eğitim)
  Early:    2016-2017 (az drift)
  Late:     2018      (test)

Kullanım:
  python predictor/prepare_energy.py
"""

import pandas as pd
import numpy as np
import os

INPUT    = "drift_mlops/real_data/PJME_hourly.csv"
OUT_DIR  = "drift_mlops/real_data"

print("=" * 60)
print("  PJME Energy Consumption - Hazırlama")
print("=" * 60)

# ── Yükle ─────────────────────────────────────────────────────────
print("\nYükleniyor...")
df = pd.read_csv(INPUT)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
df = df.dropna()

print(f"Toplam: {len(df):,} satır")
print(f"Tarih aralığı: {df['Datetime'].min()} → {df['Datetime'].max()}")
print(f"Enerji aralığı: {df['PJME_MW'].min():,.0f} - {df['PJME_MW'].max():,.0f} MW")

# ── Feature Engineering ───────────────────────────────────────────
print("\nFeature'lar oluşturuluyor...")

df['hour']       = df['Datetime'].dt.hour
df['dayofweek']  = df['Datetime'].dt.dayofweek
df['month']      = df['Datetime'].dt.month
df['quarter']    = df['Datetime'].dt.quarter
df['year']       = df['Datetime'].dt.year
df['dayofyear']  = df['Datetime'].dt.dayofyear
df['weekofyear'] = df['Datetime'].dt.isocalendar().week.astype(int)

# Binary feature'lar
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
df['is_peak']    = ((df['hour'] >= 7) & (df['hour'] <= 21)).astype(int)

# Mevsim (Kuzey Yarımküre)
def get_season(month):
    if month in [12, 1, 2]:  return 0  # Kış
    elif month in [3, 4, 5]: return 1  # İlkbahar
    elif month in [6, 7, 8]: return 2  # Yaz
    else:                     return 3  # Sonbahar

df['season'] = df['month'].apply(get_season)

# Döngüsel encoding — saat ve ay için
df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24).round(4)
df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24).round(4)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).round(4)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).round(4)

# Target
df['target'] = df['PJME_MW']

# Datetime'ı kaldır
df = df.drop(columns=['Datetime', 'PJME_MW'])

print(f"Feature'lar: {list(df.columns)}")
print(f"Toplam feature sayısı: {len(df.columns) - 1}")

# ── Yıllara göre böl ──────────────────────────────────────────────
print("\nYıllara göre bölünüyor...")

ref   = df[df['year'] <= 2015].copy()
early = df[(df['year'] == 2016) | (df['year'] == 2017)].copy()
late  = df[df['year'] == 2018].copy()

print(f"Referans (2002-2015): {len(ref):,} satır | "
      f"Ort tüketim: {ref['target'].mean():,.0f} MW")
print(f"Early    (2016-2017): {len(early):,} satır | "
      f"Ort tüketim: {early['target'].mean():,.0f} MW")
print(f"Late     (2018):      {len(late):,} satır | "
      f"Ort tüketim: {late['target'].mean():,.0f} MW")

# Kaydet
ref.to_csv(f"{OUT_DIR}/energy_reference.csv",  index=False)
early.to_csv(f"{OUT_DIR}/energy_new_early.csv", index=False)
late.to_csv(f"{OUT_DIR}/energy_new_late.csv",   index=False)

print(f"\n{'='*60}")
print("  Kaydedildi:")
print("  energy_reference.csv   (2002-2015, tam veri)")
print("  energy_new_early.csv   (2016-2017, tam veri)")
print("  energy_new_late.csv    (2018, tam veri)")
print()
print("  Target: PJME_MW (megawatt)")
print(f"{'='*60}")