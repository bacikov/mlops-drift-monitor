import pandas as pd
import numpy as np

df = pd.read_csv('drift_mlops/real_data/weatherAUS.csv')

print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Target:', df['RainTomorrow'].value_counts().to_dict())
print('Date range:', df['Date'].min(), '-', df['Date'].max())
print('Missing values:')
print(df.isnull().sum()[df.isnull().sum() > 0])
