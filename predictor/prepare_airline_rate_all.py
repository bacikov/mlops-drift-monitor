"""
Airline Rate Forecasting - Cok Yillik Aggregate v2
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "drift_mlops/real_data"
YEARS    = list(range(2009, 2019))

def process_year(year):
    path = os.path.join(DATA_DIR, f"{year}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {year}.csv not found")
        return None
    print(f"  Loading {year}.csv...")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False,
                             usecols=['FL_DATE','OP_CARRIER','ORIGIN','DEST',
                                      'CRS_DEP_TIME','CRS_ELAPSED_TIME',
                                      'DISTANCE','ARR_DELAY','CANCELLED','DIVERTED']):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    df['FL_DATE']     = pd.to_datetime(df['FL_DATE'])
    df['week_number'] = df['FL_DATE'].dt.isocalendar().week.astype(int)
    df['year']        = year
    df['dep_hour']    = (df['CRS_DEP_TIME'] // 100).clip(0, 23)
    df['is_delayed']  = (df['ARR_DELAY'] >= 15).astype(int)

    weekly = df.groupby(['year','week_number']).agg(
        n_flights         = ('FL_DATE',          'count'),
        avg_distance      = ('DISTANCE',         'mean'),
        avg_planned_time  = ('CRS_ELAPSED_TIME', 'mean'),
        morning_ratio     = ('dep_hour',         lambda x: (x.between(6,11)).mean()),
        evening_ratio     = ('dep_hour',         lambda x: (x.between(18,23)).mean()),
        long_haul_ratio   = ('DISTANCE',         lambda x: (x > 1500).mean()),
        n_carriers        = ('OP_CARRIER',       'nunique'),
        top_carrier_share = ('OP_CARRIER',       lambda x: x.value_counts(normalize=True).iloc[0]),
        n_airports        = ('ORIGIN',           'nunique'),
        delay_rate        = ('is_delayed',       'mean'),
    ).reset_index()

    weekly = weekly.sort_values('week_number')
    weekly['prev_week_delay']   = weekly['delay_rate'].shift(1).fillna(weekly['delay_rate'].mean()).round(4)
    weekly['week_sin']          = np.sin(2 * np.pi * weekly['week_number'] / 52).round(4)
    weekly['week_cos']          = np.cos(2 * np.pi * weekly['week_number'] / 52).round(4)
    weekly['is_summer']         = weekly['week_number'].between(22, 35).astype(int)
    weekly['is_holiday_season'] = weekly['week_number'].isin([1,2,47,48,49,50,51,52]).astype(int)
    weekly['is_spring']         = weekly['week_number'].between(13, 21).astype(int)
    weekly['is_fall']           = weekly['week_number'].between(36, 46).astype(int)

    print(f"  {year}: {len(weekly)} hafta | Ort: {weekly['delay_rate'].mean()*100:.1f}%")
    return weekly

print("=" * 60)
print("  Airline Rate Forecasting - 10 Yil Aggregate v2")
print("=" * 60)

all_years = []
for year in YEARS:
    r = process_year(year)
    if r is not None:
        all_years.append(r)

full  = pd.concat(all_years, ignore_index=True).sort_values(['year','week_number']).reset_index(drop=True)
train = full[full['year'] <= 2016].copy()
val   = full[full['year'] == 2017].copy()
test  = full[full['year'] == 2018].copy()

print(f"\nToplam: {len(full)} hafta")
print(f"Egitim:     {len(train)} hafta | Ort: {train['delay_rate'].mean()*100:.1f}%")
print(f"Validation: {len(val)} hafta | Ort: {val['delay_rate'].mean()*100:.1f}%")
print(f"Test:       {len(test)} hafta | Ort: {test['delay_rate'].mean()*100:.1f}%")
print(f"Feature'lar: {list(full.columns)}")

train.to_csv(os.path.join(DATA_DIR, "airline_rate_train.csv"),  index=False)
val.to_csv(os.path.join(DATA_DIR,   "airline_rate_val.csv"),    index=False)
test.to_csv(os.path.join(DATA_DIR,  "airline_rate_test.csv"),   index=False)
full.to_csv(os.path.join(DATA_DIR,  "airline_rate_all.csv"),    index=False)
print("\nKaydedildi!")
