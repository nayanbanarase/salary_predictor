# Salary Prediction ML App (Clean Version)

import pandas as pd
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
# Missing Values Handling
# ----------------------------
for col in ['Age', 'Years of Experience', 'Salary']:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

for col in ['Gender', 'Education Level', 'Job Title']:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

st.subheader("✅ Missing Values After Cleaning")
st.write(df.isnull().sum())

# ----------------------------
# Encoding Categorical Data
# ----------------------------
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Education Level', 'Job Title']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

st.subheader("🔢 Encoded Data Sample")
st.dataframe(df.head())

# ----------------------------
# Features & Target
# ----------------------------
X = df.drop('Salary', axis=1)
y = df['Salary']

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
    rmse = mse ** 0.5
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
# Save Best Model (Random Forest)
# ----------------------------
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

model_filename = "salary_model.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(best_model, file)

st.success("✅ Model trained and saved as salary_model.pkl")

st.info("✔ App is ready for deployment")
