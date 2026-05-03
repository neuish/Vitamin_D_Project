#  Vitamin D Deficiency Prediction

A machine learning project focused on predicting Vitamin D deficiency using physiological, lifestyle, and environmental factors. The goal is to identify at-risk individuals early and support preventive healthcare decisions.

---

##  Overview

Vitamin D deficiency is a widespread but often overlooked health issue. This project builds and compares multiple machine learning models to:

* Predict deficiency risk
* Analyze key contributing factors
* Provide interpretable insights using SHAP
* Enable interactive predictions via a Streamlit app

---

##  Project Structure

The main notebook: `Vitamin_D_Prediction.ipynb`

### Key Sections:

* **Problem Definition** – Context and objectives
* **Data Exploration** – Initial inspection and summary statistics
* **Data Cleaning** – Handling missing values, duplicates, and formatting
* **EDA (Visualization)** – Understanding distributions and relationships
* **Feature Engineering** – Age groups, supplement usage, encoding
* **Scaling & Normalization** – Preparing data for modeling
* **Modeling**:

  * Logistic Regression (with RFE & VIF)
  * XGBoost (GridSearch tuning)
  * CatBoost (early stopping)
  * TabNet (deep learning approach)
* **Model Comparison** – Performance benchmarking + ensemble
* **Decision Curve Analysis** – Clinical usefulness evaluation
* **SHAP Analysis** – Model interpretability
* **Deployment** – Streamlit-based prediction interface

---

##  Setup

```bash
git clone <repository_url>
cd Vitamin-D-Prediction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dataset

Place `Vitamin_D_Dataset.csv` in the root directory.

---

##  Run the App

```bash
streamlit run app.py
```

Then open the local URL in your browser to interact with:

* Predictions
* Feature inputs
* SHAP explanations

---

##  Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.838    | 0.774     | 0.849  | 0.810    | 0.926   |
| XGBoost             | 0.841    | 0.837     | 0.756  | 0.794    | 0.921   |
| CatBoost            | 0.848    | 0.830     | 0.786  | 0.807    | 0.923   |
| TabNet              | 0.802    | 0.700     | 0.896  | 0.786    | 0.907   |
| Ensemble (LR+XGB)   | 0.844    | 0.789     | 0.841  | 0.814    | 0.925   |

###  Key Insight

* Logistic Regression offers strong interpretability
* CatBoost delivers the best overall accuracy
* Ensemble balances precision and recall effectively

---

## Explainability

The project integrates **SHAP (SHapley Additive exPlanations)** to:

* Identify feature importance
* Explain individual predictions
* Improve model transparency for real-world use

---

## Deployment

A Streamlit app provides:

* User-friendly input interface
* Real-time predictions
* Visual explanation of model decisions

---

## Disclaimer

This project is for educational and research purposes only. It is not a substitute for professional medical advice or diagnosis.

---

## Future Improvements

* Add more diverse datasets
* Improve model generalization
* Deploy as a public web app
* Integrate real-time health data APIs


