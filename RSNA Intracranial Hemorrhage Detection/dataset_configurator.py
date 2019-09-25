import pandas as pd
import numpy as np

ds = pd.read_csv('stage_1_train_nice.csv')
epidurals_ds = ds[ds['epidural'] == 1].sample(100)
ds = ds.drop(epidurals_ds.index)
none_ds = ds[ds['any'] == 0].sample(100)
ds = ds.drop(none_ds.index)
epidural_train_ds = pd.concat([epidurals_ds, none_ds]).sample(frac=1)
epidural_train_ds.to_csv('epidural_train_100.csv', index=None, header=True)

epidurals_ds = ds[ds['epidural'] == 1].sample(100)
ds = ds.drop(epidurals_ds.index)
none_ds = ds[ds['any'] == 0].sample(100)
ds = ds.drop(none_ds.index)
epidural_test_ds = pd.concat([epidurals_ds, none_ds]).sample(frac=1)
epidural_test_ds.to_csv('epidural_test_100.csv', index=None, header=True)