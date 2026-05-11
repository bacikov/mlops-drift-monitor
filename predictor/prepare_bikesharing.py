"""
Bike Sharing - Veri Hazırlama v4
==================================
Senaryo A: Mevsim bazlı split (aynı yıl içinde)

2011 verisi:
  Referans: İlkbahar (Mart-Mayıs, mnth=3-5)
  Summer:   Yaz      (Haziran-Ağustos, mnth=6-8)  — ground truth var
  Autumn:   Sonbahar (Eylül-Kasım, mnth=9-11)     — gelecek, ground truth yok

Drift senaryosu:
  İlkbaharda temp~0.4, kiralama~150
  Yazda    temp~0.7, kiralama~300 → dramatik drift
  Sonbaharda temp~0.5, kiralama~200 → orta seviye

Zaman sirasi:
  İlkbahar (Mart) → Yaz (Haziran) → Sonbahar (Eylül)
  Kronolojik sira korunuyor, lag anlamli.

Feature'lar (leakage yok):
  mnth, hr, holiday, weekday, workingday,
  weathersit, temp, atemp, hum, windspeed, lag_1h

Kullanim:
  python predictor/prepare_bikesharing.py
"""

import pandas as pd
import numpy as np
import os

INPUT   = "drift_mlops/real_data/hour.csv"
OUT_DIR = "drift_mlops/real_data"

print("=" * 60)
print("  Bike Sharing - Veri Hazırlama v4")
print("  Senaryo: İlkbahar → Yaz → Sonbahar (2011)")
print("=" * 60)

df = pd.read_csv(INPUT)
df['dteday'] = pd.to_datetime(df['dteday'])
df = df.sort_values(['dteday', 'hr']).reset_index(drop=True)
print(f"Ham veri: {df.shape}")

# Lag feature — zaman sirasini koruyarak
df['lag_1h'] = df['cnt'].shift(1).fillna(df['cnt'].mean())

# Leakage sutunlari kaldir
drop_cols = ['instant', 'dteday', 'casual', 'registered', 'season', 'yr']
df_clean = df.drop(columns=drop_cols).rename(columns={'cnt': 'target'})

# Mevsim sutunlari icin orijinal veriyi kullan
mnth = df['mnth']
yr   = df['yr']

# Sadece 2011 (yr=0)
mask_2011 = (yr == 0)

# Bölünme
ref_mask    = mask_2011 & mnth.between(3, 5)   # İlkbahar
summer_mask = mask_2011 & mnth.between(6, 8)   # Yaz
autumn_mask = mask_2011 & mnth.between(9, 11)  # Sonbahar

ref    = df_clean[ref_mask].copy()
summer = df_clean[summer_mask].copy()
autumn = df_clean[autumn_mask].copy()

print(f"\nReferans  (İlkbahar, Mart-Mayıs):    {len(ref):,} saat")
print(f"  Ort kiralama: {ref['target'].mean():.0f} | Ort temp: {ref['temp'].mean():.2f}")
print(f"  Ort nem:      {ref['hum'].mean():.2f}   | Ort hava: {ref['weathersit'].mean():.2f}")

print(f"\nStream-1  (Yaz, Haz-Ağu):            {len(summer):,} saat  ← ground truth VAR")
print(f"  Ort kiralama: {summer['target'].mean():.0f} | Ort temp: {summer['temp'].mean():.2f}")

print(f"\nStream-2  (Sonbahar, Eyl-Kas):       {len(autumn):,} saat  ← ground truth YOK")
print(f"  Ort kiralama: {autumn['target'].mean():.0f} | Ort temp: {autumn['temp'].mean():.2f}")

# Baseline MAE hesapla
train_mean = ref['target'].mean()
pers_mae_ref  = float(np.mean(np.abs(ref['target'].values - ref['lag_1h'].values)))
mean_mae_ref  = float(np.mean(np.abs(ref['target'].values - train_mean)))
pers_mae_sum  = float(np.mean(np.abs(summer['target'].values - summer['lag_1h'].values)))
mean_mae_sum  = float(np.mean(np.abs(summer['target'].values - train_mean)))

print(f"\nBaseline MAE:")
print(f"  Referans (İlkbahar):")
print(f"    Persistence: {pers_mae_ref:.1f} bikes/hour")
print(f"    Mean:        {mean_mae_ref:.1f} bikes/hour")
print(f"  Yaz (ground truth):")
print(f"    Persistence: {pers_mae_sum:.1f} bikes/hour")
print(f"    Mean:        {mean_mae_sum:.1f} bikes/hour")
print(f"  (ML model yaz için persistence'ı geçmeli)")

# Kaydet
ref.to_csv(f"{OUT_DIR}/bike_reference.csv",   index=False)
summer.to_csv(f"{OUT_DIR}/bike_new_early.csv", index=False)
autumn.to_csv(f"{OUT_DIR}/bike_new_late.csv",  index=False)

print(f"\n{'='*60}")
print("  Kaydedildi:")
print("  bike_reference.csv    (İlkbahar, 2011 Mart-Mayıs)")
print("  bike_new_early.csv    (Yaz,       2011 Haz-Ağu)")
print("  bike_new_late.csv     (Sonbahar,  2011 Eyl-Kas)")
print(f"{'='*60}")
print()
print("  Drift hikayesi:")
print(f"  İlkbahar: {ref['target'].mean():.0f} bikes/hour, temp={ref['temp'].mean():.2f}")
print(f"  Yaz:      {summer['target'].mean():.0f} bikes/hour, temp={summer['temp'].mean():.2f} ← dramatik artış")
print(f"  Sonbahar: {autumn['target'].mean():.0f} bikes/hour, temp={autumn['temp'].mean():.2f}")