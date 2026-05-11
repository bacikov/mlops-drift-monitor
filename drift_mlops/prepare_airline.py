"""
Airline Delay 2018 - Ay Bazli Hazirlama
2018 verisini mevsimsel drift icin 3'e boler.

Referans:  Ocak-Nisan   (kis/bahar)
New early: Mayis-Agustos (yaz, yogun sezon)
New late:  Eylul-Aralik  (sonbahar/kis)

Hedef: ARR_DELAY >= 15 (FAA standardi)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "drift_mlops/real_data"


def clean_df(df):
    if 'CANCELLED' in df.columns:
        df = df[df['CANCELLED'] == 0].copy()
        df = df.drop('CANCELLED', axis=1)
    if 'DIVERTED' in df.columns:
        df = df[df['DIVERTED'] == 0].copy()
        df = df.drop('DIVERTED', axis=1)

    if 'ARR_DELAY' not in df.columns:
        return None

    df['target'] = (df['ARR_DELAY'] >= 15).astype(int)

    drop_cols = ['FL_DATE', 'OP_CARRIER_FL_NUM', 'WHEELS_OFF', 'WHEELS_ON',
                 'TAXI_OUT', 'TAXI_IN', 'DEP_TIME', 'ARR_TIME', 'ARR_DELAY',
                 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'CARRIER_DELAY',
                 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY',
                 'LATE_AIRCRAFT_DELAY', 'CANCELLATION_CODE', 'Unnamed: 27', 'month']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if 'OP_CARRIER' in df.columns:
        codes = {c: i for i, c in enumerate(sorted(df['OP_CARRIER'].unique()))}
        df['carrier_code'] = df['OP_CARRIER'].map(codes).fillna(-1).astype(int)
        df = df.drop('OP_CARRIER', axis=1)

    for col in ['ORIGIN', 'DEST']:
        if col in df.columns:
            freq = df[col].value_counts()
            df[col.lower() + '_freq'] = df[col].map(freq).fillna(0)
            df = df.drop(col, axis=1)

    if 'CRS_DEP_TIME' in df.columns:
        df['dep_hour'] = (df['CRS_DEP_TIME'] // 100).clip(0, 23)
        df = df.drop('CRS_DEP_TIME', axis=1)

    if 'CRS_ARR_TIME' in df.columns:
        df['arr_hour'] = (df['CRS_ARR_TIME'] // 100).clip(0, 23)
        df = df.drop('CRS_ARR_TIME', axis=1)

    if 'DEP_DELAY' in df.columns:
        df['DEP_DELAY'] = df['DEP_DELAY'].fillna(0)

    df = df.select_dtypes(include=[np.number])
    df = df.fillna(df.median())

    cols = [c for c in df.columns if c != 'target'] + ['target']
    df = df[cols]
    return df


def main():
    print("=" * 60)
    print("  Airline Delay 2018 - Mevsimsel Bolme")
    print("=" * 60)

    path = os.path.join(DATA_DIR, "2018.csv")
    if not os.path.exists(path):
        print("ERROR: 2018.csv bulunamadi!")
        return

    print("Yukleniyor...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Toplam: {len(df):,} satir")

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df['month'] = df['FL_DATE'].dt.month

    ref_df   = df[df['month'].isin([1, 2, 3, 4])].copy()
    early_df = df[df['month'].isin([5, 6, 7, 8])].copy()
    late_df  = df[df['month'].isin([9, 10, 11, 12])].copy()

    for label, part, filename in [
        ("Referans (Oca-Nis)", ref_df,   "airline_reference.csv"),
        ("Early (May-Agu)",    early_df, "airline_new_early.csv"),
        ("Late (Eyl-Ara)",     late_df,  "airline_new_late.csv"),
    ]:
        cleaned = clean_df(part)
        if cleaned is not None:
            out = os.path.join(DATA_DIR, filename)
            cleaned.to_csv(out, index=False)
            delay = cleaned['target'].mean() * 100
            size = os.path.getsize(out) / 1024 / 1024
            print(f"{label}: {len(cleaned):,} satir | Gecikme: {delay:.1f}% | {size:.0f} MB")

    print("\nHazir! Target: 'target' (1 = 15+ dk gecikme)")


if __name__ == "__main__":
    main()