"""
Credit Card Default Dataset - Veri Hazırlama
=============================================
UCI Credit Card Default (Taiwan, 2005) verisini drift tespiti için hazırlar.

Drift senaryosu:
  Referans: Genç müşteriler (AGE 20-35) - kredi geçmişi kısa
  Early:    Orta yaşlı müşteriler (AGE 35-50)
  Late:     Kıdemli müşteriler (AGE 50+)

Gerçek iş senaryosu:
  Genç müşterilerin temerrüt örüntüsü (ani nakit sıkıntısı, yüksek harcama)
  ile yaşlı müşterilerin temerrüt örüntüsü (işsizlik, emeklilik, sağlık)
  tamamen farklıdır. Bu concept drift.

Alternatif: AGE yerine kredi limitine göre de bölünebilir.
"""

import pandas as pd
import numpy as np

INPUT = 'drift_mlops/real_data/credit_default.xls'
OUTPUT_DIR = 'drift_mlops/real_data'

df = pd.read_excel(INPUT, header=1)
print(f'Ham veri: {df.shape}')

# ID gereksiz
df = df.drop('ID', axis=1)

# Hedef sütunu yeniden adlandır
df = df.rename(columns={'default payment next month': 'target'})

# Eksik değer kontrolü
print(f'Eksik değer: {df.isnull().sum().sum()}')

# Normalize et - büyük tutarları ölçeklendir
amt_cols = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
            'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

for col in amt_cols:
    df[col] = df[col] / df[col].max()

print(f'Temizlenmiş veri: {df.shape}')
print(f'Sütunlar: {list(df.columns)}')
print(f'AGE dağılımı: min={df["AGE"].min()}, max={df["AGE"].max()}, mean={df["AGE"].mean():.1f}')
print(f'Target: {df["target"].value_counts().to_dict()}')

# AGE'e göre böl
ref   = df[df['AGE'] <= 35].copy()
early = df[(df['AGE'] > 35) & (df['AGE'] <= 50)].copy()
late  = df[df['AGE'] > 50].copy()

print(f'\nReferans (20-35 yaş): {len(ref):,} satır | Default: {ref["target"].mean()*100:.1f}%')
print(f'Early (35-50 yaş):    {len(early):,} satır | Default: {early["target"].mean()*100:.1f}%')
print(f'Late (50+ yaş):       {len(late):,} satır | Default: {late["target"].mean()*100:.1f}%')

ref.to_csv(f'{OUTPUT_DIR}/credit_reference.csv', index=False)
early.to_csv(f'{OUTPUT_DIR}/credit_new_early.csv', index=False)
late.to_csv(f'{OUTPUT_DIR}/credit_new_late.csv', index=False)

print(f'\nKaydedildi:')
print(f'  credit_reference.csv  ({len(ref)} satır)')
print(f'  credit_new_early.csv  ({len(early)} satır)')
print(f'  credit_new_late.csv   ({len(late)} satır)')
print(f'  Hedef sütun: target (1=temerrüt, 0=ödedi)')
