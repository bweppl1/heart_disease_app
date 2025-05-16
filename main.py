import pandas as pd
from dataclasses import dataclass
from joblib import load

# Optimized model
model = load("heart_disease_predictor.joblib")

@dataclass
class Patient:
    age: int
    sex: int          # 1 = male, 0 = female
    cp: int           # Chest pain type (1-3)
    trestbps: int     # Resting blood pressure (mm Hg)
    chol: int         # Serum cholesterol (mg/dl)
    fbs: int          # Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no)
    restecg: int      # Resting ECG results (0-2)
    thalach: int      # Maximum heart rate achieved
    exang: int        # Exercise-induced angina (1 = yes, 0 = no)
    oldpeak: int      # ST depression induced by exercise
    slope: int        # Slope of peak exercise ST segment (0-2)
    ca: int           # Number of major vessels (0-3)
    thal: int         # Thalassemia (3,6,7)

    def to_dict(self):
        """Convert Patient object to dictionary for CSV/prediction."""
        return {
            'Age': self.age,
            'Sex': self.sex,
            'Chest pain type': self.cp,
            'BP': self.trestbps,
            'Cholesterol': self.chol,
            'FBS over 120': self.fbs,
            'EKG results': self.restecg,
            'Max HR': self.thalach,
            'Exercise angina': self.exang,
            'ST depression': self.oldpeak,
            'Slope of ST': self.slope,
            'Number of vessels fluro': self.ca,
            'Thallium': self.thal
        }
def main():
    print("Welcome to the Unofficial Heart Disease Predictor!")
    patient = collect_patient_data()
    save_and_predict(patient)

# New patient input UI
def collect_patient_data():
    print("\nNEW PATIENT ENTRY")
    print("=" * 20)
    
    age = int(input("Age: "))
    sex = int(input("Sex (1 = male, 0 = female): "))
    cp = int(input("Chest pain type (1 = typical, 2 = atypical, 3 = non-anginal, 4 = asymptomatic): "))
    trestbps = int(input("Resting blood pressure (mm Hg): "))
    chol = int(input("Serum cholesterol (mg/dl): "))
    fbs = int(input("Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no): "))
    restecg = int(input("Resting ECG results (0-2): "))
    thalach = int(input("Maximum heart rate achieved: "))
    exang = int(input("Exercise-induced angina (1 = yes, 0 = no): "))
    oldpeak = int(input("ST depression induced by exercise(1 = yes, 0 = no): "))
    slope = int(input("Slope of peak exercise ST segment (1 = upslope, 2 = flat, 3 = downslope): "))
    ca = int(input("Number of major vessels (0-3): "))
    thal = int(input("Thalassemia (3-normal,6-fixed defect,7-reversible defect): "))
    
    return Patient(age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                  exang, oldpeak, slope, ca, thal)

# New patient info prediction, return as % chance with predict_proba
def predict_heart_disease(patient):
    patient_df = pd.DataFrame([patient.to_dict()])
    proba = model.predict_proba(patient_df)[:, 1]
    return proba[0]

# Save and send data for prediction
def save_and_predict(patient, filename="data/new_patient_data.csv"):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=patient.to_dict().keys())
    
    # Append and save
    df = pd.concat([df, pd.DataFrame([patient.to_dict()])], ignore_index=True)
    df.to_csv(filename, index=False)
    
    # Predict
    risk = predict_heart_disease(patient)
    print(f"\nPredicted heart disease risk: {risk:.1%}")
    return risk

if __name__ == "__main__":
    main()