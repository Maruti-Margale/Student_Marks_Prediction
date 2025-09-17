import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

# Load model
with open("Student_Marks_Prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit config
st.set_page_config(page_title="Student Marks Predictor", layout="centered")

# Custom CSS for background and buttons
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #004080;
            color: white;
        }
        .stButton > button:hover {
            background-color: #0066cc;
        }
    </style>
""", unsafe_allow_html=True)

# --- Navigation Bar ---
with st.sidebar:
    selected = option_menu(
        menu_title="📘 Menu",
        options=["Home", "About"],
        icons=["house", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#004080"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#0066cc"},
        }
    )

# --- Home Page ---
if selected == "Home":
    st.title("🎯 Student Marks Predictor")

    st.markdown("Enter student data to predict their expected marks:")

    # Custom feature inputs based on your training
    number_courses = st.number_input("📚 Number of Courses", min_value=1, step=1)
    time_study = st.number_input("⏱️ Time Spent Studying (in hours)", min_value=0.0, step=0.5)

    if st.button("🎓 Predict Marks"):
        input_array = np.array([[number_courses, time_study]])
        try:
            prediction = model.predict(input_array)[0]
            st.success(f"✅ Predicted Marks: **{prediction:.2f}** out of 100")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# --- About Page ---
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This app predicts student marks based on:
        - 📚 Number of courses
        - ⏱️ Time spent studying
        
        It uses a machine learning model trained with **Random Forest**, **Gradient Boosting**, and **AdaBoost** regressors.

        ✅ Built with: **Python, Scikit-learn, Streamlit**
    """)
    st.markdown("---")
    st.info("You can extend this app by adding charts, feature importance, or uploading CSVs.")

# Footer
st.markdown("---")
st.caption("📘 Student Marks Predictor | Made with ❤️ using Streamlit")
