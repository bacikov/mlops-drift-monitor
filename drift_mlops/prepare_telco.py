"""
Telco Customer Churn - Veri Hazırlama
======================================
IBM Telco Customer Churn verisini drift tespiti için hazırlar.

Drift senaryosu:
  Referans: Kısa süreli müşteriler (tenure 0-24 ay) - yeni müşteriler
  Early:    Orta süreli müşteriler (tenure 24-48 ay)
  Late:     Uzun süreli müşteriler (tenure 48+ ay)

Gerçek bir iş senaryosu:
  Yeni müşteriler üzerinde eğitilen model eski müşterilere uygulandığında
  churn örüntüsü farklıdır — uzun süreli müşteriler farklı sebeplerle
  churn eder (fiyat artışı, rakip teklif, hizmet kalitesi).
"""

import pandas as pd
import numpy as np

INPUT = 'drift_mlops/real_data/telco_churn.csv'
OUTPUT_DIR = 'drift_mlops/real_data'

df = pd.read_csv(INPUT)
print(f'Ham veri: {df.shape}')

# customerID gereksiz
df = df.drop('customerID', axis=1)

# TotalCharges bazen string geliyor
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Hedef sütun: Churn → 0/1
df['target'] = (df['Churn'] == 'Yes').astype(int)
df = df.drop('Churn', axis=1)

# Kategorik sütunları encode et
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies']

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
                               'No phone service': 0, 'No internet service': 0})

# Contract: Month-to-month=0, One year=1, Two year=2
df['Contract'] = df['Contract'].map({
    'Month-to-month': 0, 'One year': 1, 'Two year': 2
})

# InternetService: No=0, DSL=1, Fiber optic=2
df['InternetService'] = df['InternetService'].map({
    'No': 0, 'DSL': 1, 'Fiber optic': 2
})

# PaymentMethod → sayısal
df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3,
})

# Eksik kalan varsa medyan ile doldur
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna()

print(f'Temizlenmiş veri: {df.shape}')
print(f'Sütunlar: {list(df.columns)}')
print(f'Target: {df["target"].value_counts().to_dict()}')
print(f'Tenure dağılımı: min={df["tenure"].min()}, max={df["tenure"].max()}, mean={df["tenure"].mean():.1f}')

# Tenure'a göre böl
# Referans: 0-24 ay (yeni müşteriler)
# Early:    24-48 ay (orta vadeli)
# Late:     48+ ay (uzun vadeli)

ref   = df[df['tenure'] <= 24].copy()
early = df[(df['tenure'] > 24) & (df['tenure'] <= 48)].copy()
late  = df[df['tenure'] > 48].copy()

print(f'\nReferans (0-24 ay):  {len(ref):,} satır | Churn: {ref["target"].mean()*100:.1f}%')
print(f'Early (24-48 ay):    {len(early):,} satır | Churn: {early["target"].mean()*100:.1f}%')
print(f'Late (48+ ay):       {len(late):,} satır | Churn: {late["target"].mean()*100:.1f}%')

ref.to_csv(f'{OUTPUT_DIR}/telco_reference.csv', index=False)
early.to_csv(f'{OUTPUT_DIR}/telco_new_early.csv', index=False)
late.to_csv(f'{OUTPUT_DIR}/telco_new_late.csv', index=False)

print(f'\nKaydedildi:')
print(f'  telco_reference.csv  ({len(ref)} satır)')
print(f'  telco_new_early.csv  ({len(early)} satır)')
print(f'  telco_new_late.csv   ({len(late)} satır)')
print(f'  Hedef sütun: target (1=churn, 0=kaldı)')
