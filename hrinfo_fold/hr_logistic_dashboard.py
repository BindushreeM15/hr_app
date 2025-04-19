# hr_logistic_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

st.set_page_config(page_title="HR Exit Predictor", layout="wide")
st.title(" HR Dashboard to Predict the Exit of Employees")

# Upload dataset
uploaded_file = st.file_uploader("Upload HR dataset (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Encode for filter (not model)
    df['Department'] = df['Department'].astype(str)
    df['salary'] = df['salary'].astype(str)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Sidebar Filters
    st.sidebar.header("🔍 Filter Options")
    dept = st.sidebar.selectbox("Select Department", options=["All"] + sorted(df['Department'].unique()))
    salary = st.sidebar.selectbox("Select Salary Level", options=["All"] + sorted(df['salary'].unique()))

    filtered_df = df.copy()
    if dept != "All":
        filtered_df = filtered_df[filtered_df['Department'] == dept]
    if salary != "All":
        filtered_df = filtered_df[filtered_df['salary'] == salary]

    # Scatter plot
    st.subheader("📈 Satisfaction vs Average Monthly Hours (Filtered)")
    if 'left' in filtered_df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df, x='satisfaction_level', y='average_montly_hours', hue='left', ax=ax)
        st.pyplot(fig)

    # 🎯 Prediction Form
    st.subheader("🎯 Predict if a New Employee May Exit")

    with st.form("prediction_form"):
        satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
        projects = st.number_input("Number of Projects", 1, 10, 3)
        avg_hours = st.number_input("Average Monthly Hours", 50, 400, 160)
        years = st.number_input("Time Spent at Company", 1, 10, 3)
        accident = st.selectbox("Work Accident", [0, 1])
        promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
        department = st.selectbox("Department", sorted(df['Department'].unique()))
        salary_input = st.selectbox("Salary Level", ['low', 'medium', 'high'])

        # Match encoding used in training
        dept_map = {name: i for i, name in enumerate(sorted(df['Department'].unique()))}
        salary_map = {'low': 0, 'medium': 1, 'high': 2}

        dept_num = dept_map[department]
        salary_num = salary_map[salary_input]

        submitted = st.form_submit_button("Predict Exit")

    if submitted:
        model = pickle.load(open("logreg_model.pkl", "rb"))
        input_data = [[
            satisfaction, evaluation, projects, avg_hours,
            years, accident, promotion, dept_num, salary_num
        ]]
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # probability of leaving

        if prediction == 1:
            st.error(f"⚠️ The employee is likely to leave. (Prob: {prob:.2f})")
        else:
            st.success(f"✅ The employee is likely to stay. (Prob: {1 - prob:.2f})")

        # Show feature importance
        # st.subheader("📌 Feature Importance (Logistic Coefficients)")
        # features = ['satisfaction_level', 'last_evaluation', 'number_project',
        #             'average_montly_hours', 'time_spend_company', 'Work_accident',
        #             'promotion_last_5years', 'Department', 'salary']
        # coefs = model.coef_[0]
        # importance_df = pd.DataFrame({
        #     'Feature': features,
        #     'Coefficient': coefs,
        #     'Absolute Importance': np.abs(coefs)
        # }).sort_values(by='Absolute Importance', ascending=True)

        # fig2, ax2 = plt.subplots(figsize=(10, 6))
        # ax2.barh(importance_df['Feature'], importance_df['Absolute Importance'], color='lightgreen')
        # ax2.set_xlabel("Absolute Coefficient Value")
        # ax2.set_title("Feature Importance (Logistic Regression)")
        # st.pyplot(fig2)

else:
    st.info("📂 Please upload your HR dataset (CSV) to begin.")
