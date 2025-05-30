### Project Overview

This is a ML/AI program to predict the presence of heart disease in patients. The model uses a Random Forest Classifier, optimized with StratifiedKFold and RandomizedSearchCV. Due to the use-case of the model, the algorithm is optimized to minimize false negatives.

This program was created and tested using VSCode V1.99 on Linux Mint.

### Try It Out!

The model is deployed on streamlit, use the link below to try it out!

https://heartdiseaseapp-gbjfbr6jmifilrxuk7mcrr.streamlit.app/

### C.1. Requirements

#### Software

- VSCode 1.99
- Python 3.10.12
- Git (for cloning repository, version control)

#### Libraries

- matplotlib==3.10.1
- seaborn==0.13.2
- joblib==1.4.2
- pandas==2.2.3
- numpy==2.2.5
- scikit-learn==1.6.1
- streamlit==1.45.1

#### Hardware

- Minimum of 4GB RAM Recommended

### C.2. Instructions

1. Clone the program repository:

```bash
git clone https://github.com/bweppl1/heart_disease_app.git
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Use the trained model to make predictions on new patients:

```bash
python optimized_model.py
```
