import pandas as pd
import numpy as np

df = pd.read_csv('drift_mlops/real_data/weatherAUS.csv')

# Tarih sütununu isle
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Cok eksik olan sutunlari kaldir
drop_cols = ['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm',
             'WindGustDir', 'WindDir9am', 'WindDir3pm']
df = df.drop(columns=drop_cols)

# RainToday ve RainTomorrow: Yes/No -> 1/0
df['RainToday'] = (df['RainToday'] == 'Yes').astype(int)
df['RainTomorrow'] = (df['RainTomorrow'] == 'Yes').astype(int)

# Hedef sutunu yeniden adlandir
df = df.rename(columns={'RainTomorrow': 'target'})

# Eksik degerleri sutun ortalamasi ile doldur
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Hedef eksikse kaldir
df = df.dropna(subset=['target'])

print('Temizlenmis veri:', df.shape)
print('Sutunlar:', list(df.columns))
print('Target:', df['target'].value_counts().to_dict())
print('Month distribution:')
print(df['Month'].value_counts().sort_index())

# Mevsime gore bol
# Avustralya'da: Yaz = Aralik-Subat, Kis = Haziran-Agustos
# Referans: Yaz aylari (Aralik, Ocak, Subat) - sicak, az yagmur
# Early drift: Ilkbahar/Sonbahar (Mart-May, Eyl-Kas)
# Late drift: Kis (Haziran-Agustos) - soguk, cok yagmur

ref   = df[df['Month'].isin([12, 1, 2])].drop('Month', axis=1)    # Yaz
early = df[df['Month'].isin([3, 4, 5, 9, 10, 11])].drop('Month', axis=1)  # Ilkbahar/Sonbahar
late  = df[df['Month'].isin([6, 7, 8])].drop('Month', axis=1)     # Kis

print()
print('Referans (Yaz):          ', len(ref), 'satir | Yagmur:', ref['target'].sum(), f"({ref['target'].mean()*100:.1f}%)")
print('Early drift (Ilkbahar):  ', len(early), 'satir | Yagmur:', early['target'].sum(), f"({early['target'].mean()*100:.1f}%)")
print('Late drift (Kis):        ', len(late), 'satir | Yagmur:', late['target'].sum(), f"({late['target'].mean()*100:.1f}%)")

ref.to_csv('drift_mlops/real_data/weather_reference.csv', index=False)
early.to_csv('drift_mlops/real_data/weather_new_early.csv', index=False)
late.to_csv('drift_mlops/real_data/weather_new_late.csv', index=False)

print()
print('Kaydedildi:')
print('  weather_reference.csv')
print('  weather_new_early.csv')
print('  weather_new_late.csv')
print('  Hedef sutun: target')
