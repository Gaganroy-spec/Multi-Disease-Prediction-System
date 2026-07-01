# data_generator.py
import numpy as np
import pandas as pd

# ------------------- Fever Dataset -------------------
def generate_fever_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    # Features
    temperature = np.random.normal(98.6, 1.5, n_samples)
    cough = np.random.binomial(1, 0.3, n_samples)
    headache = np.random.binomial(1, 0.2, n_samples)
    fatigue = np.random.binomial(1, 0.25, n_samples)
    sore_throat = np.random.binomial(1, 0.15, n_samples)
    muscle_ache = np.random.binomial(1, 0.2, n_samples)
    
    # Target: fever = 1 if temperature > 100.4 OR (cough+headache+fatigue >=2 and temperature>99.5)
    fever = ((temperature > 100.4) | 
             ((cough + headache + fatigue >= 2) & (temperature > 99.5))).astype(int)
    
    # Add some noise (flip 5% of labels)
    noise_idx = np.random.choice(n_samples, size=int(0.05*n_samples), replace=False)
    fever[noise_idx] = 1 - fever[noise_idx]
    
    df = pd.DataFrame({
        'temperature': temperature,
        'cough': cough,
        'headache': headache,
        'fatigue': fatigue,
        'sore_throat': sore_throat,
        'muscle_ache': muscle_ache,
        'fever': fever
    })
    return df

# ------------------- Diabetes Dataset -------------------
def generate_diabetes_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    pregnancies = np.random.randint(0, 17, n_samples)
    glucose = np.random.normal(120, 30, n_samples)
    blood_pressure = np.random.normal(70, 12, n_samples)
    skin_thickness = np.random.normal(25, 8, n_samples)
    insulin = np.random.normal(80, 40, n_samples)
    bmi = np.random.normal(28, 6, n_samples)
    dpf = np.random.uniform(0.1, 2.5, n_samples)   # diabetes pedigree function
    age = np.random.normal(35, 12, n_samples)
    
    # Synthetic target: diabetes if (glucose>140 and bmi>30) or (age>50 and dpf>1.5) etc.
    diabetes = ((glucose > 140) & (bmi > 30)) | ((age > 50) & (dpf > 1.5)) | (pregnancies > 5)
    diabetes = diabetes.astype(int)
    
    # Add noise
    noise_idx = np.random.choice(n_samples, size=int(0.05*n_samples), replace=False)
    diabetes[noise_idx] = 1 - diabetes[noise_idx]
    
    df = pd.DataFrame({
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': dpf,
        'age': age,
        'diabetes': diabetes
    })
    return df

# ------------------- Heart Attack Dataset -------------------
def generate_heart_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    age = np.random.normal(55, 12, n_samples)
    sex = np.random.binomial(1, 0.68, n_samples)   # 1=male
    cp = np.random.choice([0,1,2,3], n_samples, p=[0.4,0.3,0.2,0.1])   # chest pain type
    trestbps = np.random.normal(130, 20, n_samples)   # resting blood pressure
    chol = np.random.normal(240, 50, n_samples)       # cholesterol
    fbs = np.random.binomial(1, 0.15, n_samples)      # fasting blood sugar >120
    restecg = np.random.choice([0,1,2], n_samples, p=[0.5,0.4,0.1])
    thalach = np.random.normal(150, 25, n_samples)    # max heart rate
    exang = np.random.binomial(1, 0.3, n_samples)     # exercise induced angina
    oldpeak = np.random.exponential(1.0, n_samples)   # ST depression
    slope = np.random.choice([0,1,2], n_samples, p=[0.4,0.4,0.2])
    ca = np.random.choice([0,1,2,3], n_samples, p=[0.7,0.2,0.07,0.03])
    thal = np.random.choice([0,1,2,3], n_samples, p=[0.3,0.4,0.2,0.1])
    
    # Risk score (higher = more risk)
    risk = (0.03*age + 0.5*sex + 0.8*(cp>1) + 0.01*trestbps + 0.01*chol + 
            1.2*fbs + 0.5*(restecg>0) - 0.02*thalach + 1.5*exang + 1.2*oldpeak +
            0.7*(slope>0) + 1.2*ca + 0.8*(thal>1))
    prob = 1 / (1 + np.exp(-(risk - 5)))   # logistic transform
    target = (prob > 0.5).astype(int)
    
    # add noise
    noise_idx = np.random.choice(n_samples, size=int(0.05*n_samples), replace=False)
    target[noise_idx] = 1 - target[noise_idx]
    
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
        'heart_attack': target
    })
    return df

if __name__ == "__main__":
    fever_df = generate_fever_data()
    fever_df.to_csv("fever_data.csv", index=False)
    print("Saved fever_data.csv")
    
    diabetes_df = generate_diabetes_data()
    diabetes_df.to_csv("diabetes_data.csv", index=False)
    print("Saved diabetes_data.csv")
    
    heart_df = generate_heart_data()
    heart_df.to_csv("heart_data.csv", index=False)
    print("Saved heart_data.csv")