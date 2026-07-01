# model_test.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_condition(condition_name):
    # Load model and test data
    model = joblib.load(f"{condition_name}_best_model.pkl")
    X_test = pd.read_csv(f"{condition_name}_X_test.csv")
    y_test = pd.read_csv(f"{condition_name}_y_test.csv").squeeze()   # convert to Series
    
    y_pred = model.predict(X_test)
    
    print(f"\n========== {condition_name.upper()} ==========")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_condition('fever')
    evaluate_condition('diabetes')
    evaluate_condition('heart')