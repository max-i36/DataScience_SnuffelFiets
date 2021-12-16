import pandas as pd

df = pd.read_csv('snuffelfiets_data.csv')
df.drop(['air_quality_observed_id',
         'geom',
         'recording_time',
         'voc',
         'acc_max',
         'no2',
         'horizontal_accuracy',
         'vertical_accuracy',
         ],
        axis=1,
        inplace=True,
        )

df.to_csv('snuffelfiets_data_filtered.csv')
