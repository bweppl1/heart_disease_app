import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd

#Read in CSV File
df = pd.read_csv('data/Heart_Disease_Prediction.csv')

#Viewing data layout
df.info()

#B1 Data Preprocessing
#Convert categorical variables, data cleaning, handle missing values -> Upload preprocessed dataset to GitLab
#Converting Absence/Presnce to Binary Presence Flag
df['Heart Disease'] = (df['Heart Disease'] == 'Presence').astype(int)

df['Heart Disease'].value_counts() # 0 = Absence of HD; 1 = Presene of HD

#Confirming null value count
df.isnull().sum()

#Save cleaned, preprocessed dataframe
df.to_csv('data/Cleaned_Heart_Disease_Prediction.csv', index=False)
