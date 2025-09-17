import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("Student_Marks_Prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("📘 Student Marks Predictor")
st.write("Enter the number of study hours to predict the expected marks.")

# User input
hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)

# Prediction
if st.button("Predict Marks"):
    input_data = np.array(hours).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    st.success(f"📚 Predicted Marks: **{prediction:.2f}** out of 100")

# Footer
st.markdown("---")
st.caption("Created using Streamlit | Model: Student_Marks_Prediction.pkl")
