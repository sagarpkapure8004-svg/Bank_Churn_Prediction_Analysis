import streamlit as st
import pandas as pd
import joblib

# This loads your specific file
model = joblib.load('random_forest_model_compressed.pkl')
# 1. Page Configuration
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

# --- 1. Sidebar Inputs (Top-Level) ---

MODEL_FEATURES = list(getattr(model, 'feature_names_in_', []))


def _parse_numeric_bin(col_name: str, prefix: str):
    value_text = col_name[len(prefix):].strip()
    value_text = value_text.strip(' ()')
    if value_text.endswith('%'):
        value_text = value_text[:-1]
    value_text = value_text.strip().lstrip('_').strip()
    try:
        return float(value_text)
    except ValueError:
        return None


def _set_closest_bin(prefix: str, value: float, row: dict):
    candidates = [c for c in MODEL_FEATURES if c.startswith(prefix)]
    if not candidates or value is None:
        return
    best = min(
        candidates,
        key=lambda c: abs((_parse_numeric_bin(c, prefix) or 0.0) - value)
    )
    best_value = _parse_numeric_bin(best, prefix)
    if best_value is None:
        return
    for candidate in candidates:
        if _parse_numeric_bin(candidate, prefix) == best_value:
            row[candidate] = 1


def build_model_input(
    geography,
    gender,
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_crcard,
    is_active,
    estimated_salary,
    engagement_score,
):
    row = dict.fromkeys(MODEL_FEATURES, 0)

    if 'France ' in row:
        row['France '] = 1 if geography == 'France' else 0
    if 'Spain ' in row:
        row['Spain '] = 1 if geography == 'Spain' else 0
    if 'Male' in row:
        row['Male'] = 1 if gender == 'Male' else 0

    if 'CreditScore' in row:
        row['CreditScore'] = credit_score
    if 'NumOfProducts' in row:
        row['NumOfProducts'] = num_products
    if 'Product to tenure Ratio ' in row:
        row['Product to tenure Ratio '] = num_products / (tenure + 1)
    if 'Age Tenure' in row:
        row['Age Tenure'] = age * tenure

    b_s_ratio = balance / (estimated_salary + 1)
    scaled_score = credit_score / 850
    scaled_age = age / 100
    scaled_tenure = tenure / 10
    scaled_balance = balance / 250000
    scaled_salary = estimated_salary / 100000
    engagement_pct = engagement_score * 10
    bsr_pct = b_s_ratio * 100

    _set_closest_bin('Scaled Score_', scaled_score, row)
    _set_closest_bin('Scaled Age_', scaled_age, row)
    _set_closest_bin('Scaled Tenure_', scaled_tenure, row)
    _set_closest_bin('Scaled Balance _', scaled_balance, row)
    _set_closest_bin('Scaled Salary_', scaled_salary, row)
    _set_closest_bin('Engagement Score_', engagement_pct, row)
    _set_closest_bin('Balance to Salary Ratio_', bsr_pct, row)

    return row


def user_input_features():
    st.sidebar.header("User Input Features")

    geography = st.sidebar.selectbox("Geography", options=("France", "Germany", "Spain"))
    gender = st.sidebar.selectbox("Gender", options=("Male", "Female"))
    credit_score = st.sidebar.number_input("Credit Score", 300, 850, 600)
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 75000.0)
    num_products = st.sidebar.selectbox("Number of Products", options=(1, 2, 3, 4))
    has_crcard = st.sidebar.selectbox("Has Credit Card?", options=(1, 0))
    is_active = st.sidebar.selectbox("Is Active Member?", options=(1, 0))
    estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 75000.0)
    engagement_score = st.sidebar.slider("Customer Engagement Score", 0.0, 10.0, 5.0)

    summary = {
        'Geography': geography,
        'Gender': gender,
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_crcard,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Engagement Score': engagement_score,
    }

    model_input = build_model_input(
        geography,
        gender,
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_crcard,
        is_active,
        estimated_salary,
        engagement_score,
    )

    return pd.DataFrame(summary, index=[0]), pd.DataFrame([model_input]), geography, gender
# --- 2. Execution ---
summary_df, model_input_df, geo_val, gen_val = user_input_features()

# Create display version for the table
display_df = summary_df.copy()


# --- 3. Main Page UI ---
st.title("🏦 Bank Customer Churn Intelligence")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Customer Profile Summary")
    st.dataframe(display_df, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction")
    if st.button('Analyze Risk Status'):
        try:
            prediction = model.predict(model_input_df)
            if prediction[0] == 1:
                st.error("### ⚠️ High Risk")
                st.write("Customer is likely to **Churn**.")
            else:
                st.success("### ✅ Low Risk")
                st.write("Customer is likely to **Stay**.")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

# --- 4. Visual Analysis & Graphs (The Previous Model Dashboard) ---
st.markdown("---")

# 4. Adding Metric Cards at the bottom for extra 'Dashboard' feel
st.markdown("---")
st.subheader("Key Performance Indicators")
m1, m2, m3 = st.columns(3)

# Logic to make metrics dynamic
score_text = "Excellent" if summary_df['CreditScore'][0] >= 700 else "Fair"
balance_text = "High Value" if summary_df['Balance'][0] > 100000 else "Standard"

m1.metric("Credit Standing", score_text)
m2.metric("Account Tier", balance_text)
m3.metric("Product Count", f"{summary_df['NumOfProducts'][0]} Items")
# 4. Visual Analysis Section
st.markdown("---")
st.header("📊 Model Analysis & Insights")

# Using columns for the first two charts
col_img1, col_img2 = st.columns(2)

with col_img1:
    st.subheader("Feature Importance")
    # Matching your filename: shap.png
    st.image("shap.png", caption="Impact of features on prediction", use_container_width=True)

with col_img2:
    st.subheader("Model Performance Metrics")
    # Matching your filename: download (1).png
    st.image("download (1).png", caption="Confusion Matrix / Accuracy Metrics", use_container_width=True)

# Optional: SHAP explanation in an expander for a professional touch
with st.expander("ℹ️ What do these charts mean?"):
    st.write("""
        The **SHAP** chart shows which factors (like Age, Balance, or Number of Products) 
        most influenced the model's decision to predict Churn or Stay. 
        The **Performance** chart shows how accurate the model was during testing.
    """)