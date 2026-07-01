# model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ------------------- Helper Function -------------------
def train_and_save(data_path, target_col, categorical_features, numeric_features, condition_name):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save test data for later evaluation
    X_test.to_csv(f"{condition_name}_X_test.csv", index=False)
    y_test.to_csv(f"{condition_name}_y_test.csv", index=False)
    
    # Models to try
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print(f"\n===== Training models for {condition_name} =====")
    for name, model in models.items():
        # Build pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_model = pipeline
            best_name = name
    
    print(f"Best model for {condition_name}: {best_name} with accuracy {best_score:.4f}")
    # Save best pipeline
    joblib.dump(best_model, f"{condition_name}_best_model.pkl")
    print(f"Saved {condition_name}_best_model.pkl")
    
    # Optional: also save the preprocessor alone (already inside pipeline)
    return best_model

# ------------------- Fever -------------------
fever_cat = ['cough', 'headache', 'fatigue', 'sore_throat', 'muscle_ache']
fever_num = ['temperature']
train_and_save('fever_data.csv', 'fever', fever_cat, fever_num, 'fever')

# ------------------- Diabetes -------------------
# All features numeric (no encoding needed)
diabetes_cat = []   # no categorical
diabetes_num = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age']
train_and_save('diabetes_data.csv', 'diabetes', diabetes_cat, diabetes_num, 'diabetes')

# ------------------- Heart Attack -------------------
heart_cat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
heart_num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
train_and_save('heart_data.csv', 'heart_attack', heart_cat, heart_num, 'heart')