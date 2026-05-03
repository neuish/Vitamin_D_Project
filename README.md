# Vitamin D Deficiency Prediction

This project aims to develop and evaluate several machine learning models to predict Vitamin D deficiency based on various physiological, lifestyle, and environmental factors. The primary goal is to identify individuals at risk of Vitamin D deficiency, enabling early intervention and personalized health recommendations.

## Project Structure

The notebook `Vitamin_D_Prediction.ipynb` contains the full analysis, including:

1.  **Problem Statement**: Introduction to the problem of Vitamin D deficiency.
2.  **Data Loading and Describing**: Loading the dataset and performing initial data inspection.
3.  **Data Cleaning and Treating**: Handling missing values, duplicates, and renaming columns.
4.  **Visualization on Vitamin D Deficiency**: Exploratory Data Analysis (EDA) through various plots to understand data distributions and relationships.
5.  **Feature Engineering**: Creating new features like age groups and vitamin D supplement groups, and one-hot encoding categorical variables.
6.  **Scaling and Normalization**: Scaling continuous features for model compatibility.
7.  **Train-Test Split**: Dividing the data into training and testing sets.
8.  **Model Training and Evaluation**: Training and evaluating Logistic Regression, XGBoost, CatBoost, and TabNet models.
    *   **Logistic Regression**: Includes Feature Selection using RFE and VIF analysis.
    *   **XGBoost**: Hyperparameter tuning using GridSearchCV.
    *   **CatBoost**: Training with early stopping.
    *   **TabNet**: Training with early stopping.
9.  **Model Comparison**: Comparing all models based on various metrics, including a soft voting ensemble.
10. **Decision Curve Analysis**: Assessing the clinical utility of the models.
11. **SHAP Integration**: Explaining model predictions and feature importance for Logistic Regression and XGBoost.
12. **Prediction**: A Streamlit application for interactive predictions and SHAP explanations.

## Setup Instructions

To run this project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Vitamin-D-Prediction
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**: Ensure you have the `Vitamin_D_Dataset.csv` file in the root directory of the project.

    *If running in Google Colab*, the dataset is expected to be mounted from Google Drive at `/content/drive/MyDrive/Logistic Regression/Vitamin_D_Dataset.csv`.

## How to Run the Streamlit Application

The project includes a Streamlit application (`app.py` or the code integrated into the notebook's 'Prediction' section) for interactive prediction and model explanation. To run the Streamlit app locally:

1.  **Ensure you have `app.py` and `Vitamin_D_Dataset.csv` in the same directory.**
2.  **Run the Streamlit application** from your terminal in the project's root directory:
    ```bash
    streamlit run app.py
    ```

This will open the application in your web browser, allowing you to interact with the prediction interface and view model explanations.

## Model Evaluation Summary

Various classification models were trained and evaluated:

| Model                 | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 0.838    | 0.774     | 0.849  | 0.810    | 0.926   |
| XGBoost               | 0.841    | 0.837     | 0.756  | 0.794    | 0.921   |
| CatBoost              | 0.848    | 0.830     | 0.786  | 0.807    | 0.923   |
| TabNet                | 0.802    | 0.700     | 0.896  | 0.786    | 0.907   |
| Ensemble (LR+XGB)     | 0.844    | 0.789     | 0.841  | 0.814    | 0.925   |

The models show competitive performance, with Logistic Regression, XGBoost, and CatBoost achieving high ROC-AUC scores, indicating strong discriminatory power. The ensemble model combines the strengths of Logistic Regression and XGBoost, offering a balanced performance.
