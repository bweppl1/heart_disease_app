#Loading optimized model for use with new data in the future
from joblib import load

model = load("heart_disease_predictor.joblib")

