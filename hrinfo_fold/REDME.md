# 👩‍💼 HR to predict exit of employees  – Streamlit App

A simple and interactive HR analytics dashboard built with **Streamlit**, which predicts whether an employee is likely to **stay or leave** based on key HR factors using a **Logistic Regression model**.

---

## 🚀 Features

- Upload and explore your own HR dataset (CSV)
- Filter by department and salary level
- Visualize satisfaction vs. monthly hours
- Predict if a new employee may exit
- Clean and user-friendly interface

---

## 🧠 Model Info

The trained model uses the following features:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `promotion_last_5years`
- `Department` (label encoded)
- `salary` (label encoded)

Model used: **Logistic Regression**  
Accuracy: **~75.8%** (based on train-test split)

---


