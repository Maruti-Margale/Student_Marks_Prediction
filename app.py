import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

# Load model
with open("Student_Marks_Prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Try to get expected number of input features
try:
    n_features = model.n_features_in_
except AttributeError:
    st.error("❌ Model input feature count could not be detected.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Student Marks Predictor", layout="centered")

# --- Custom Styling ---
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

# --- Home Page (Main App) ---
if selected == "Home":
    st.title("🎯 Student Marks Predictor")

    st.markdown("Enter the values for the features used to train the model:")

    input_values = []
    for i in range(n_features):
        val = st.number_input(f"Feature {i+1}", step=1.0, format="%.2f")
        input_values.append(val)

    if st.button("🎓 Predict Marks"):
        input_array = np.array(input_values).reshape(1, -1)
        try:
            prediction = model.predict(input_array)[0]
            st.success(f"✅ Predicted Marks: **{prediction:.2f}** out of 100")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# --- About Page ---
elif selected == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
        This web app predicts student marks based on input features provided by the user.
        
        - Built using **Streamlit**
        - Uses a pre-trained machine learning model
        - You can deploy this app using **Streamlit Cloud** or locally

        **Developer:** Maruti Margale
    """)
    st.markdown("---")
    st.info("You can customize this About section with more content.")

# Footer
st.markdown("---")
st.caption("📘 Student Marks Predictor | Made with ❤️ using Streamlit")
