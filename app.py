
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    with open('random_forest_regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'random_forest_regressor_model.pkl' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# User inputs for features
age = st.slider('Age', 18, 65, 30)
gender_options = {'Male': 1, 'Female': 0} # Using encoded values from LabelEncoder
gender_selected = st.selectbox('Gender', list(gender_options.keys()))
gender = gender_options[gender_selected]

education_level_options = {'Bachelor\'s': 0, 'Master\'s': 1, 'PhD': 2} # Using encoded values
education_level_selected = st.selectbox('Education Level', list(education_level_options.keys()))
education_level = education_level_options[education_level_selected]

# For 'Job Title', direct input of encoded value as it's harder to map all 100+ titles
st.info("Please enter the numerical encoded value for Job Title. Refer to your training data for correct mappings.")
job_title = st.number_input('Job Title (Encoded Value)', min_value=0, max_value=200, value=100)

years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0)

if st.button('Predict Salary'):
    # Create a DataFrame for the input features
    input_data = pd.DataFrame([[age, gender, education_level, job_title, years_of_experience]],
                                columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${prediction:,.2f}')
