import streamlit as st
import numpy as np
import pandas as pd
import joblib

## Load trained model
model = joblib.load('salary_model.pk1')

## Streamlit App
st.title("Salary Prediction")

## Define the input options
year_of_enrollment= ['2021', '2022', '2023']
graduation_year= ['2023', '2024', '2025']
test_score= [round(x * 0.1, 1) for x in range(1, 101)]

## User inputs
year_of_enrollment_selected = st.selectbox("Select year of enrollement", year_of_enrollment)
graduation_year_selected = st.selectbox("Select graduation year", graduation_year)
test_score_selected = st.selectbox("Select test score", test_score)
gpa_or_score_selected = st.slider("Select gpa score", min_value=0.01,
                                  max_value=5.00, value=2.50)

##Predict button
if st.button("Predict Salary amount"):

    ## Create dict for input features
    input_data = {
        'year_of_enrollment': year_of_enrollment_selected,
        'graduation_year': graduation_year_selected,
        'test_score': test_score_selected,
        'gpa_or_score': gpa_or_score_selected
    }

    ## Convert input data to a Dataframe
    df_input = pd.DataFrame({
        'year_of_enrollment': [year_of_enrollment_selected],
        'graduation_year': [graduation_year_selected],
        'test_score': [test_score_selected],
        'gpa_or_score': [gpa_or_score_selected]
    })

    ## One-hot encoding
    df_input = pd.get_dummies(df_input,
                              columns = ['year_of_enrollment', 'graduation_year', 'test_score']
                              )
    
    # df_input = df_input.to_numpy()

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)
    
    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Salary amount: ${y_unseen_pred:,.2f}")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3Dbackgrounds&psig=AOvVaw3rAnntA_8ASbsLAcRekwHK&ust=1752118486324000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCMC4_JHsro4DFQAAAAAdAAAAABAE")
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)
