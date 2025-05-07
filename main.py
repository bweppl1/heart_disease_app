import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

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

#B6 Hyperparameter Tuning
#Hyperparameters to sample from
param_dist = {
    "n_estimators": np.arange(50, 300, 50),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": np.arange(2, 11),
    "min_samples_leaf": np.arange(1, 5),
    "bootstrap": [True, False],
    "class_weight": [None, "balanced"]
}

#Randomized search on the paramater set
rand_search = RandomizedSearchCV(
    estimator=rfc_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=skf_model,
    scoring="f1",
    n_jobs=2,
    random_state=42
)

rand_search.fit(X_train, y_train)

#Printing 
print("\nBest f1 Score: ", rand_search.best_score_)
print("Best Parameters: ", rand_search.best_params_)

#Save model
optimized_model = rand_search.best_estimator_
dump(optimized_model, "heart_disease_predictor.joblib")

#Best model evaluation metrics
best_y_pred = optimized_model.predict(X_test)
best_y_proba = optimized_model.predict_proba(X_test)[:, 1]

final_metrics = {
    "Precision": precision_score(y_test, best_y_pred),
    "Recall": recall_score(y_test, best_y_pred),
    "F1-Score": f1_score(y_test, best_y_pred),
    "AUC ROC": roc_auc_score(y_test, best_y_proba)
}

#Display best model metrics
print("\nBest Model Metrics")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.5f}")

#Visualize model comparison(Base vs. StratifiedKFold, vs. RandomizedSearchCV)
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

#Visualizing comparisons and results
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
plt.title("StratifiedKFold Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

#RandomizedSearchCV bar plot
plt.figure(figsize=(5,4))
sns.barplot(data=final_metrics)
plt.title("RandomizedSearchCV Performance")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()

#Classification Report
print("\nClassification Report")
print(classification_report(y_test, best_y_pred, target_names=["No Disease", "Disease"]))

#Confusion Matrix
cm = confusion_matrix(y_test, best_y_pred)

cv_cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No HD", "HD"])
cv_cm_display.plot(cmap="Blues", colorbar=False)
plt.title("CV Confusion Matrix")
plt.show()