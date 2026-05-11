import pandas as pd

df = pd.read_csv('drift_mlops/real_data/creditcard.csv')
df = df.rename(columns={'Class': 'target'})

# Zamana gore bol
ref   = df[df['Time'] < 86400].drop('Time', axis=1)
early = df[(df['Time'] >= 86400) & (df['Time'] < 129600)].drop('Time', axis=1)
late  = df[df['Time'] >= 129600].drop('Time', axis=1)

print('Referans:', len(ref), 'satir | Fraud:', ref['target'].sum())
print('Early:   ', len(early), 'satir | Fraud:', early['target'].sum())
print('Late:    ', len(late), 'satir | Fraud:', late['target'].sum())

ref.to_csv('drift_mlops/real_data/creditcard_reference.csv', index=False)
early.to_csv('drift_mlops/real_data/creditcard_new_early.csv', index=False)
late.to_csv('drift_mlops/real_data/creditcard_new_late.csv', index=False)
print('Kaydedildi.')
