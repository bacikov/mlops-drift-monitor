import pandas as pd

files = [
    ('airline_reference', 'airline_reference_sample'),
    ('airline_new_early', 'airline_new_early_sample'),
    ('airline_new_late', 'airline_new_late_sample'),
]

for src, dst in files:
    df = pd.read_csv('drift_mlops/real_data/' + src + '.csv')
    sample = df.sample(n=100000, random_state=42)
    sample.to_csv('drift_mlops/real_data/' + dst + '.csv', index=False)
    delay = sample['target'].mean() * 100
    print(dst + ': ' + str(len(sample)) + ' satir | Gecikme: ' + str(round(delay, 1)) + '%')
