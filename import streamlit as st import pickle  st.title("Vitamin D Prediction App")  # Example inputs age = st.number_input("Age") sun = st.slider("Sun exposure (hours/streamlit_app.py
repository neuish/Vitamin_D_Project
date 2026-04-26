import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout="wide")
st.title("Vitamin D Deficiency Analysis Dashboard")

# -----------------------------
# LOAD DATA (IMPORTANT FIX)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")  # put your dataset in repo

df = load_data()

# -----------------------------
# COLORS
# -----------------------------
BG = '#0b0f1a'
PANEL = '#111827'
TEAL = '#22d3ee'
CORAL = '#f87171'
GOLD = '#f5c842'

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Model Prediction"])

# -----------------------------
# EDA PAGE
# -----------------------------
if page == "EDA":

    st.header("Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.histplot(df["vitamin_d_ng_ml"], kde=True, ax=ax)
    ax.axvline(20, color=GOLD, linestyle="--")
    ax.set_title("Vitamin D Distribution")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x="sun_hours_per_day",
        y="vitamin_d_ng_ml",
        hue="deficient",
        data=df,
        ax=ax2
    )
    ax2.axhline(20, color=GOLD, linestyle="--")
    st.pyplot(fig2)

    st.write("Dataset shape:", df.shape)

# -----------------------------
# MODEL PAGE
# -----------------------------
elif page == "Model Prediction":

    st.header("Vitamin D Prediction")

    # Load trained model (IMPORTANT FIX)
    model = pickle.load(open("model.pkl", "rb"))

    age = st.number_input("Age", 1, 100)
    sun = st.slider("Sun exposure (hours/day)", 0, 12)
    bmi = st.number_input("BMI", 10, 50)

    if st.button("Predict"):
        input_data = [[age, sun, bmi]]
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("High Risk of Vitamin D Deficiency")
        else:
            st.success("Low Risk of Vitamin D Deficiency")
