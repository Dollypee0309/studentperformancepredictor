import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os 

# Load model
rf_model = joblib.load("student_model.pkl")

# User login info
usernames = {"admin": "password123", "teacher": "abc123"}

# Page config
st.set_page_config(page_title="Student Prediction Dashboard", layout="centered")

# Session state init
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Page navigation buttons
st.markdown("""
<style>
    .nav-btn {
        display: inline-block;
        margin-right: 10px;
        padding: 8px 16px;
        background-color: #0366d6;
        color: white;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Login"):
        st.session_state.page = "Login"
with col2:
    if st.button("Input Student Data"):
        st.session_state.page = "Input"
with col3:
    if st.button("View Prediction & Intervention"):
        st.session_state.page = "Predict"
with col4:
    if st.button("View All Saved Students Data"):
        st.session_state.page = "ViewData"

# Save to CSV
def save_input_to_csv(data, filename="student_inputs.csv"):
    columns = ["Age", "Gender", "Ethnicity", "Socioeconomic_Status", "Parental_Education",
               "GPA", "Past_Academic_Performance", "Current_Semester_Performance",
               "Courses_Failed", "Credits_Completed", "Study_Hours_per_Week", "Attendance_Rate",
               "Late_Submissions", "Class_Participation_Score", "Online_Learning_Hours",
               "Library_Usage_Hours", "Disciplinary_Actions", "Social_Engagement_Score",
               "Mental_Health_Score", "Extracurricular_Activities"]
    df = pd.DataFrame([data], columns=columns)

    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

# Login page
if st.session_state.page == "Login":
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login!"):
        if username in usernames and usernames[username] == password:
            st.session_state.logged_in = True
            st.session_state.page = "Input"
            st.success("Login successful! Redirecting to Input Page...")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Input page
elif st.session_state.page == "Input":
    if st.session_state.logged_in:
        st.title("Student Performance Data Entry")

        st.session_state.age = st.slider("Age", 15, 40, 20)
        st.session_state.gender = st.selectbox("Gender", ["Male", "Female"])
        st.session_state.ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"])
        st.session_state.ses = st.selectbox("Socioeconomic Status", ["Low", "Medium", "High"])
        st.session_state.parent_edu = st.selectbox("Parental Education", ["High School", "Bachelor’s", "Master’s"])
        st.session_state.gpa = st.number_input("GPA", 0.0, 5.0)
        st.session_state.past_perf = st.selectbox("Past Academic Performance", ["Below Average", "Average", "Above Average"])
        st.session_state.current_perf = st.slider("Current Semester Performance", 0, 100)
        st.session_state.courses_failed = st.number_input("Courses Failed", 0, 10)
        st.session_state.credits = st.number_input("Credits Completed", 0, 150)
        st.session_state.hours_study = st.number_input("Study Hours/Week", 0, 50)
        st.session_state.attendance = st.slider("Attendance Rate (%)", 0, 100)
        st.session_state.late_submissions = st.slider("Late Submissions", 0, 10)
        st.session_state.participation = st.slider("Class Participation Score", 0, 10)
        st.session_state.online_hours = st.slider("Online Learning Hours", 0, 30)
        st.session_state.library_hours = st.slider("Library Usage Hours", 0, 20)
        st.session_state.discipline = st.slider("Disciplinary Actions", 0, 5)
        st.session_state.social = st.slider("Social Engagement Score", 0, 10)
        st.session_state.mental_health = st.slider("Mental Health Score", 0, 10)
        st.session_state.extracurricular = st.slider("Extracurricular Activities", 0, 5)

        if st.button("Submit Student Data"):
            raw_input = [st.session_state.age, st.session_state.gender, st.session_state.ethnicity, st.session_state.ses,
                         st.session_state.parent_edu, st.session_state.gpa, st.session_state.past_perf, st.session_state.current_perf,
                         st.session_state.courses_failed, st.session_state.credits, st.session_state.hours_study, st.session_state.attendance,
                         st.session_state.late_submissions, st.session_state.participation, st.session_state.online_hours,
                         st.session_state.library_hours, st.session_state.discipline, st.session_state.social,
                         st.session_state.mental_health, st.session_state.extracurricular]
            save_input_to_csv(raw_input)
            st.success("✅ Student data saved! Redirecting to Prediction Page...")
            
            st.session_state.page = "Predict"
            st.rerun()
    else:
        st.warning("Please log in first.")

# Prediction page
elif st.session_state.page == "Predict":
    if st.session_state.logged_in:
        st.title("Student Performance Prediction")

        required_keys = [
            "age", "gender", "ethnicity", "ses", "parent_edu", "gpa", "past_perf", "current_perf",
            "courses_failed", "credits", "hours_study", "attendance", "late_submissions", "participation",
            "online_hours", "library_hours", "discipline", "social", "mental_health", "extracurricular"
        ]

        if all(key in st.session_state and st.session_state[key] is not None for key in required_keys):
            # Encode categorical features
            gender_val = 1 if st.session_state.gender == "Female" else 0
            eth_val = {"Asian":0, "Black":1, "Hispanic":2, "White":3, "Other":4}[st.session_state.ethnicity]
            ses_val = {"Low":0, "Medium":1, "High":2}[st.session_state.ses]
            edu_val = {"High School":0, "Bachelor’s":1, "Master’s":2}[st.session_state.parent_edu]
            past_perf_val = {"Below Average":0, "Average":1, "Above Average":2}[st.session_state.past_perf]


            input_data = np.array([[st.session_state.age, gender_val, eth_val, ses_val, edu_val, st.session_state.gpa,
                                    past_perf_val, st.session_state.current_perf, st.session_state.courses_failed, st.session_state.credits,
                                    st.session_state.hours_study, st.session_state.attendance, st.session_state.late_submissions, st.session_state.participation,
                                    st.session_state.online_hours, st.session_state.library_hours, st.session_state.discipline, st.session_state.social,
                                    st.session_state.mental_health, st.session_state.extracurricular, 0]])  # Dummy value as 21st feature


            if st.button("Predict Dropout Risk"):
                prediction = rf_model.predict(input_data)[0]
                label = "At Risk" if prediction == 1 else "Not at Risk"

                st.subheader(f"Prediction Result: {label}")

                if prediction == 1:
                    st.info("Suggested Interventions:")
                    st.markdown("- Offer academic counseling")
                    st.markdown("- Assign a mentor")
                    st.markdown("- Monitor attendance closely")
                    st.markdown("- Mental health check-up recommended")

            if st.button("Go Back to Input Page"):
                st.session_state.page = "Input"
        else:
            st.warning("Incomplete data. Please return to 'Input Student Data' Page and complete all fields.")
    else:
        st.warning("Please log in first.")

#view saved students data
elif st.session_state.page == "ViewData":
    if st.session_state.logged_in:
        st.title("Saved Student Record")
        if os.path.exists("student_inputs.csv"):
            df = pd.read_csv("student_inputs.csv")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "student_records.csv", "text/csv")
        else:
            st.info("No student data found yet.")
    else:
        st.warning("Please log in first.")