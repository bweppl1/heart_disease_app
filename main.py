import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

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

#Set 'Heart Disease' as the target column
#B2 Build the algorithm
X = df.drop(columns=['Heart Disease'])
y = df['Heart Disease']

#Split data to training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#B3 Assigning and train model
rfc_model = RandomForestClassifier(random_state=42)
rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)

#B4 Evaluate model accuracy
#Evaluation metrics - f1score, precision, recall, auc-roc
rfc_model_f1_score = f1_score(y_test, y_pred)
rfc_precision = precision_score(y_test, y_pred)
rfc_recall = recall_score(y_test, y_pred)
rfc_auc_roc = roc_auc_score(y_test, y_pred)

#Display Base Metrics
print("Base Model Evaluation Metrics")
print("Precision: {:.5f}".format(rfc_precision))
print("Recall: {:.5f}".format(rfc_recall))
print("AUC ROC: {:.5f}".format(rfc_auc_roc))
print("F1 Score: {:.5f}".format(rfc_model_f1_score))