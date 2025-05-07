import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

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
rfc_f1_score = f1_score(y_test, y_pred)
rfc_precision = precision_score(y_test, y_pred)
rfc_recall = recall_score(y_test, y_pred)
rfc_auc_roc = roc_auc_score(y_test, y_pred)

#Display Base Metrics
print("\nBase Model Evaluation Metrics")
print("Precision: {:.5f}".format(rfc_precision))
print("Recall: {:.5f}".format(rfc_recall))
print("AUC ROC: {:.5f}".format(rfc_auc_roc))
print("F1 Score: {:.5f}".format(rfc_f1_score))

#B5 Cross-validation
skf_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    rfc_model,
    X,
    y,
    cv=skf_model,
    scoring=["precision", "recall", "f1", "roc_auc"]
)

cv_predictions = cross_val_predict(rfc_model, X, y, cv=skf_model, method='predict')

cv_f1_score = f1_score(y, cv_predictions)
cv_precision = precision_score(y, cv_predictions)
cv_recall = recall_score(y, cv_predictions)
cv_auc_roc = roc_auc_score(y, cv_predictions)

print("\nCross Validation Evaluation Metrics")
print("CV Precision: {:.5F}".format(cv_precision))
print("CV Recall: {:.5F}".format(cv_recall))
print("CV ROC AUC: {:.5F}".format(cv_auc_roc))
print("CV F1 Score: {:.5F}".format(cv_f1_score))

#Visualize pre vs. post cross validation
base_metrics = {
    "Precision": rfc_precision,
    "Recall": rfc_recall,
    "F1-Score": rfc_f1_score,
    "AUC ROC": rfc_auc_roc
}

cv_metrics = {
    "Precision": cv_precision,
    "Recall": cv_recall,
    "F1-Score": cv_f1_score,
    "AUC ROC": cv_auc_roc 
}

#Base Random Forest Classifier bar plot
plt.figure(figsize=(5,4))
sns.barplot(data=base_metrics, )
plt.title("Base Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

#CV Metrics bar plot
plt.figure(figsize=(5,4))
sns.barplot(data=cv_metrics)
plt.title("Cross Validation Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()