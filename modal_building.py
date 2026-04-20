# Salary Prediction ML App (Fixed Version)

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("💼 Salary Prediction ML App")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(
    'https://raw.githubusercontent.com/nayanbanarase/salary_predictor/refs/heads/main/Salary_Data.csv'
)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Clean column names (important fix)
# ----------------------------
df.columns = df.columns.str.strip()

# ----------------------------
# Replace inf with NaN
# ----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ----------------------------
# Fill missing values
# ----------------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------
# Encoding categorical columns
# ----------------------------
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

st.subheader("🔢 Encoded Data Sample")
st.dataframe(df.head())

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop('Salary', axis=1)
y = df['Salary']

# ----------------------------
# Final safety check (VERY IMPORTANT)
# ----------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Models
# ----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, mse, rmse, r2])

# ----------------------------
# Results Table
# ----------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "RMSE", "R2"]
)

st.subheader("📈 Model Performance Comparison")
st.dataframe(results_df.sort_values(by="R2", ascending=False))

# ----------------------------
# Train Best Model
# ----------------------------
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

# ----------------------------
# Save Model
# ----------------------------
with open("salary_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

st.success("✅ Model trained and saved successfully as salary_model.pkl")
st.info("✔ App is ready for deployment")
