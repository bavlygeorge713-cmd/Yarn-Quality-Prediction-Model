import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\bavly\OneDrive\Desktop\smart system\yarn_training_data_10000.csv"
data = pd.read_csv(file_path)
print(f"Shape: {data.shape}")
print("Columns:", list(data.columns))
print("Sample rows:", data.head(), "\n")

target_cols = [
    "yarn_strength",
    "yarn_elongation",
    "yarn_cv",
    "yarn_thin_places",
    "yarn_thick_places",
    "yarn_neps",
    "yarn_hairiness",
]

X = data.drop(columns=["blend_id"] + target_cols)  # every column but features
y = data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

for target in target_cols:
    print(f" Training model for: {target}")

    rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    rf.fit(X_train_scaled, y_train[target])

    gb = GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=5, random_state=42
    )
    gb.fit(X_train_scaled, y_train[target])

    preds_rf = rf.predict(X_test_scaled)
    preds_gb = gb.predict(X_test_scaled)
    preds_final = (
        preds_rf + preds_gb
    ) / 2  # Simple averaging ensemble between RF and GBM
    # Evaluate model
    r2 = r2_score(y_test[target], preds_final)
    mae = mean_absolute_error(y_test[target], preds_final)

    results[target] = (r2, mae)

print(" Model Performance Summary:")
for t, (r2, mae) in results.items():
    print(f"{t:20s} | R² = {r2:6.4f} | MAE = {mae:10.4f}")

sample = X_test.iloc[[0]]
sample_scaled = scaler.transform(sample)
sample_preds = {}

for target in target_cols:
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    gb = GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=5, random_state=42
    )
    rf.fit(X_train_scaled, y_train[target])
    gb.fit(X_train_scaled, y_train[target])
    sample_preds[target] = (
        (rf.predict(sample_scaled) + gb.predict(sample_scaled)) / 2
    )[0]

print(" Example prediction for one random blend:")
print(pd.DataFrame([sample_preds]))
