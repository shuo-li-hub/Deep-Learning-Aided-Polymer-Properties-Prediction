import numpy as np
import pandas as pd
targets=['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_list = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
extra_Tc=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
extra_FFV=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')
print(train_list[targets].count())

extra_Tc=extra_Tc[['TC_mean','SMILES']].rename(columns={'TC_mean':'Tc'})
extra_Tc[['id','Tg','FFV','Density','Rg']]=float('nan')
extra_FFV[['id','Tg','Tc','Density','Rg']]=float('nan')

extra_Tc = extra_Tc[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
extra_FFV = extra_FFV[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
display(extra_Tc.head(2))
display(extra_FFV.head(2))

train_list=pd.concat([train_list,extra_Tc,extra_FFV],ignore_index=True)
print(train_list[targets].count())
