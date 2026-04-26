%%writefile dashboard_app.py

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming df is loaded globally or passed here
# For the purpose of this Streamlit app, we will assume df is available or loaded within the app.
# In a real scenario, you would load your data inside the Streamlit app.

# Global theme from notebook (re-define for standalone app)
BG      = '#0b0f1a'
PANEL   = '#111827'
WHITE   = '#f8fafc'
MUTED   = '#64748b'
GOLD    = '#f5c842'
TEAL    = '#22d3ee'
CORAL   = '#f87171'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': PANEL,
    'text.color': WHITE,
    'axes.labelcolor': MUTED,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.grid': True
})

# --- Define plotting functions for reusability ---

def plot_vitamin_d_distribution(ax, df_data, GOLD_color, TEAL_color, CORAL_color):
    sns.kdeplot(data=df_data, x='vitamin_d_ng_ml', hue='deficient', fill=True, palette={0: TEAL_color, 1: CORAL_color}, ax=ax)
    ax.axvline(20, color=GOLD_color, linestyle='--')
    ax.set_title('Vitamin D Distribution & Threshold')
    ax.set_xlabel('Vitamin D (ng/mL)')
    ax.set_ylabel('Density')

def plot_sun_exposure_vs_vitamin_d(ax, df_data, GOLD_color, TEAL_color, CORAL_color):
    sns.scatterplot(x='sun_hours_per_day', y='vitamin_d_ng_ml', hue='deficient', data=df_data, palette={0: TEAL_color, 1: CORAL_color}, alpha=0.6, ax=ax)
    ax.axhline(20, linestyle='--', color=GOLD_color)
    ax.set_title('Sun Exposure vs Vitamin D')
    ax.set_xlabel('Sun Hours Per Day')
    ax.set_ylabel('Vitamin D (ng/mL)')

def plot_bmi_vs_vitamin_d(ax, df_data, GOLD_color):
    df_copy = df_data.copy()
    df_copy['bmi_cat'] = pd.cut(df_copy['bmi'], bins=[0,18.5,25,30,100], labels=['Underweight','Normal','Overweight','Obese'])
    sns.violinplot(x='bmi_cat', y='vitamin_d_ng_ml', data=df_copy, ax=ax, palette='viridis', hue='bmi_cat', legend=False)
    ax.axhline(20, linestyle='--', color=GOLD_color)
    ax.set_title('BMI vs Vitamin D')
    ax.set_xlabel('BMI Category')
    ax.set_ylabel('Vitamin D (ng/mL)')
    ax.tick_params(axis='x', rotation=30)

def plot_season_vs_vitamin_d(ax, df_data, GOLD_color):
    sns.boxplot(x='season', y='vitamin_d_ng_ml', data=df_data, ax=ax, palette='viridis', hue='season', legend=False)
    ax.axhline(20, linestyle='--', color=GOLD_color)
    ax.set_title('Season vs Vitamin D')
    ax.set_xlabel('Season')
    ax.set_ylabel('Vitamin D (ng/mL)')

def plot_feature_importance(ax, df_data, TEAL_color):
    features = [
        'sun_hours_per_day',
        'outdoor_activity_minutes',
        'bmi',
        'latitude_deg',
        'screen_time_hours',
        'diet_score',
        'vitamin_d_supplement_iu'
    ]
    X_dashboard = df_data[features]
    y_dashboard = df_data['deficient']

    pipe_dashboard = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=500))
    ])
    pipe_dashboard.fit(X_dashboard, y_dashboard)
    coef_dashboard = pd.Series(pipe_dashboard.named_steps['lr'].coef_[0], index=features)

    coef_dashboard.sort_values().plot(kind='barh', color=TEAL_color, ax=ax)
    ax.set_title('Feature Importance (Logistic Regression)')
    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Feature')

def plot_risk_score_vs_deficiency(ax, df_data, CORAL_color):
    df_copy = df_data.copy()
    df_copy['risk_score'] = (
        (df_copy['sun_hours_per_day'] < 2).astype(int) +
        (df_copy['bmi'] > 27).astype(int) +
        (df_copy['vitamin_d_supplement_iu'] == 0).astype(int)
    )
    risk = df_copy.groupby('risk_score')['deficient'].mean()*100
    risk.plot(kind='bar', color=CORAL_color, ax=ax)
    ax.set_title('Risk Score vs Deficiency Rate')
    ax.set_xlabel('Risk Score (Number of Risk Factors)')
    ax.set_ylabel('% Deficient')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=0)

def plot_skin_tone_vs_vitamin_d(ax, df_data, GOLD_color):
    sns.violinplot(x='skin_tone', y='vitamin_d_ng_ml', data=df_data, palette='viridis', ax=ax, hue='skin_tone', legend=False)
    ax.axhline(20, color=GOLD_color, linestyle='--', label='Deficiency Threshold')
    ax.set_title('Vitamin D Levels by Skin Tone')
    ax.set_xlabel('Skin Tone')
    ax.set_ylabel('Vitamin D (ng/mL)')
    ax.legend(title='Threshold', loc='upper right', frameon=False)

def plot_deficiency_by_season_rate(ax, df_data):
    deficiency_by_season_dash = df_data.groupby('season')['deficient'].mean().reset_index()
    sns.barplot(x='season', y='deficient', data=deficiency_by_season_dash, palette='coolwarm', ax=ax, hue='season', legend=False)
    ax.set_title('Vitamin D Deficiency Rate by Season')
    ax.set_xlabel('Season')
    ax.set_ylabel('Proportion Deficient')
    ax.set_ylim(0, 1)

def plot_key_driver_correlation(ax, df_data):
    key_features_corr = ['sun_hours_per_day', 'outdoor_activity_minutes', 'bmi', 'latitude_deg', 'screen_time_hours', 'diet_score', 'vitamin_d_supplement_iu', 'deficient']
    key_corr_matrix = df_data[key_features_corr].corr()
    sns.heatmap(key_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix of Key Drivers')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

# --- Streamlit App Layout ---
st.set_page_config(layout='wide') # Use wide layout
st.title('Vitamin D Deficiency Analysis Dashboard')

# In a real Streamlit app, you would load your data here.
# For this example, we assume 'df' is loaded outside and available, or we mock it.
# Replace with your actual data loading mechanism if deploying standalone.

# Mocking df for demonstration if running directly as a script without Colab context
# In Colab, the global df from previous cells would be used.
if 'df' not in st.session_state:
    # This part would typically load your data from a CSV or other source
    # For simplicity, we'll try to use the global df if running in Colab, else a placeholder
    try:
        # Assuming df is available in the global scope from Colab execution
        st.session_state.df = globals()['df']
    except KeyError:
        st.error("DataFrame 'df' not found. Please ensure it's loaded in your environment.")
        st.stop()

df_streamlit = st.session_state.df

fig = plt.figure(figsize=(24, 28)) # Adjust figure size for optimal display
gs = gridspec.GridSpec(3, 3, figure=fig) # 3 rows, 3 columns

# Row 1
ax0 = fig.add_subplot(gs[0, 0])
plot_vitamin_d_distribution(ax0, df_streamlit, GOLD, TEAL, CORAL)

ax1 = fig.add_subplot(gs[0, 1])
plot_sun_exposure_vs_vitamin_d(ax1, df_streamlit, GOLD, TEAL, CORAL)

ax2 = fig.add_subplot(gs[0, 2])
plot_bmi_vs_vitamin_d(ax2, df_streamlit, GOLD)

# Row 2
ax3 = fig.add_subplot(gs[1, 0])
plot_season_vs_vitamin_d(ax3, df_streamlit, GOLD)

ax4 = fig.add_subplot(gs[1, 1])
plot_feature_importance(ax4, df_streamlit, TEAL)

ax5 = fig.add_subplot(gs[1, 2])
plot_risk_score_vs_deficiency(ax5, df_streamlit, CORAL)

# Row 3
ax6 = fig.add_subplot(gs[2, 0])
plot_skin_tone_vs_vitamin_d(ax6, df_streamlit, GOLD)

ax7 = fig.add_subplot(gs[2, 1])
plot_deficiency_by_season_rate(ax7, df_streamlit)

ax8 = fig.add_subplot(gs[2, 2])
plot_key_driver_correlation(ax8, df_streamlit)


plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent overlap, leaving space for suptitle
st.pyplot(fig)
