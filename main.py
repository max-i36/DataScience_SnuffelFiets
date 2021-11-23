import pandas as pd
import numpy as np

df = pd.read_csv('snuffelfiets_data.csv')

acc = df['acc_max']

# print(df)

acc_array = list(acc)

acc_array = np.array(acc_array)

acc_filtered = np.where(acc_array != 0)

print(np.max(acc_array))
print(acc_filtered[0])
print(len(acc_filtered[0]))
print(len(acc_array))
