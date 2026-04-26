import streamlit as st
import pickle

st.title("Vitamin D Prediction App")

# Example inputs
age = st.number_input("Age")
sun = st.slider("Sun exposure (hours/day)", 0, 12)

# Load model
model = pickle.load(open("model.pkl", "rb"))

if st.button("Predict"):
    prediction = model.predict([[age, sun]])
    st.write("Prediction:", prediction)
