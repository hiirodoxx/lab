import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.cache_data.clear()
st.cache_resource.clear()

## Load trained model
model = joblib.load('salary_model.pk1')

## Streamlit App
st.title("Salary Prediction")

## Define the input options
year_of_enrollment= ['2019', '2020', '2021', '2022', '2023']
graduation_year= ['2020', '2021', '2022', '2023', '2024', '2025']
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
        background: url("https://imageio.forbes.com/specials-images/imageserve/679cc6d58ccd4a393431ccc2/Growth-financial-business-arrow-money-coin-on-increase-earnings-3d-background-with/960x0.jpg?format=jpg&width=960")
        background-size: cover
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
