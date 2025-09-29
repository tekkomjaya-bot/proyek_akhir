from modules.diabetes_pred_transform import NUMERICAL_FEATURE, CATEGORICAL_FEATURES, transform_name, LABEL_KEY
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for feature in CATEGORICAL_FEATURES:
    print(f"Categorical feature: {feature}")

for feature in NUMERICAL_FEATURE:
    print(f"Numerical feature: {feature}")

print("="*50)

for name in CATEGORICAL_FEATURES:
    print(transform_name(name))

df = pd.read_csv("data/Disease_symptom_and_patient_profile_dataset.csv")

print(df[LABEL_KEY].value_counts())
print("\n")

for feature in CATEGORICAL_FEATURES:
    df[transform_name(feature)] = le.fit_transform(df[feature])
    print(f"Transformed {feature} to {df[transform_name(feature)].value_counts()}")