
import pandas as pd
import numpy as np


noise_data = pd.read_csv('noise_data.csv')
print(np.sum(pd.isna(noise_data).to_numpy()))


noise_data.replace('-16,000.00', np.nan, inplace=True)
print(np.sum(pd.isna(noise_data).to_numpy()))

