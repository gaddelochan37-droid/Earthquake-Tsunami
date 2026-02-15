import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

print("Loading earthquake dataset...")
df = pd.read_csv('data/earthquake.csv')

X = df.drop('tsunami', axis=1)
y = df['tsunami']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Save test data for later evaluation
test_df = X_test.copy()
test_df['tsunami'] = y_test
test_df.to_csv('data/test_data.csv', index=False)
print(f"[OK] Test data saved to data/test_data.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

print("\n" + "="*60)
print("Training Models...")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if name in ['Logistic Regression', 'kNN']:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    with open(f'models/{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [OK] Saved to models/{name.replace(' ', '_').lower()}.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*60)
print("All models trained and saved successfully!")
print("="*60)
print("\nModels saved:")
for name in models.keys():
    print(f"  - models/{name.replace(' ', '_').lower()}.pkl")
print(f"  - models/scaler.pkl")
print("\nNote: Evaluation metrics will be calculated when test data is uploaded via the Streamlit app.")
