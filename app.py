import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# This loads your specific file
model = joblib.load('random_forest_model_compressed.pkl')
MODEL_FEATURES = list(getattr(model, 'feature_names_in_', []))

# 1. Page Configuration
st.set_page_config(
    page_title="Bank Churn Predictor Pro",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #d32f2f;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #f57c00;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #388e3c;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def get_customer_segment(age, balance, credit_score, num_products):
    """Determine customer segment based on profile"""
    if credit_score >= 750 and balance > 100000:
        return "Premium", "🏆"
    elif credit_score >= 650 and balance > 50000:
        return "Gold", "🥇"
    elif age < 30 and num_products >= 3:
        return "Young Professional", "👨‍💼"
    elif age > 50 and balance > 75000:
        return "Senior Value", "👴"
    else:
        return "Standard", "👤"

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability > 0.7:
        return "High Risk", "🔴", "#d32f2f"
    elif probability > 0.4:
        return "Medium Risk", "🟡", "#f57c00"
    else:
        return "Low Risk", "🟢", "#388e3c"

def generate_recommendations(prediction, probability, customer_data):
    """Generate actionable recommendations based on prediction"""
    recommendations = []

    if prediction == 1:  # High churn risk
        if customer_data['IsActiveMember'][0] == 0:
            recommendations.append("📞 Schedule a personal call to re-engage the customer")
        if customer_data['NumOfProducts'][0] == 1:
            recommendations.append("🎁 Offer additional products/services with incentives")
        if customer_data['Balance'][0] < 50000:
            recommendations.append("💰 Provide balance growth incentives or rewards")
        if customer_data['Age'][0] < 35:
            recommendations.append("🎯 Target with youth-focused retention campaigns")
        recommendations.append("📧 Send personalized retention email within 24 hours")
        recommendations.append("⭐ Offer loyalty program upgrade or special benefits")
    else:  # Low churn risk
        recommendations.append("✅ Continue current engagement strategies")
        if customer_data['NumOfProducts'][0] < 3:
            recommendations.append("📈 Consider cross-selling additional products")
        recommendations.append("🎉 Recognize customer loyalty with appreciation rewards")

    return recommendations

def create_risk_gauge(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 40], 'color': "#e8f5e8"},
                {'range': [40, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_customer_profile_chart(customer_data):
    """Create a radar chart showing customer profile"""
    categories = ['Credit Score', 'Age', 'Balance', 'Tenure', 'Products', 'Engagement']

    # Normalize values for radar chart
    values = [
        customer_data['CreditScore'][0] / 850 * 100,  # Credit score as percentage
        min(customer_data['Age'][0] / 80 * 100, 100),  # Age normalized
        min(customer_data['Balance'][0] / 250000 * 100, 100),  # Balance normalized
        customer_data['Tenure'][0] / 10 * 100,  # Tenure as percentage
        customer_data['NumOfProducts'][0] / 4 * 100,  # Products as percentage
        customer_data['Engagement Score'][0] * 10  # Engagement score
    ]

    values += values[:1]  # Close the radar chart
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Customer Profile',
        line_color='#1f77b4'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        title="Customer Profile Analysis"
    )
    return fig


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
st.markdown('<h1 class="main-header">🏦 Bank Customer Churn Intelligence Pro</h1>', unsafe_allow_html=True)

# Add tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Assessment", "📊 Analytics Dashboard", "👥 Customer Insights", "📋 Batch Analysis"])

with tab1:
    st.markdown("### Customer Risk Assessment")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Customer Profile Summary")
        st.dataframe(display_df, use_container_width=True)

        # Customer segmentation
        segment, icon = get_customer_segment(
            summary_df['Age'][0],
            summary_df['Balance'][0],
            summary_df['CreditScore'][0],
            summary_df['NumOfProducts'][0]
        )
        st.markdown(f"**Customer Segment:** {icon} {segment}")

    with col2:
        st.subheader("🎯 Prediction Results")
        if st.button('🔍 Analyze Risk Status', type='primary', use_container_width=True):
            try:
                prediction = model.predict(model_input_df)
                # Get prediction probability
                prediction_proba = model.predict_proba(model_input_df)[0]
                churn_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction[0]

                risk_level, risk_icon, risk_color = get_risk_level(churn_probability)

                # Risk assessment display
                if prediction[0] == 1:
                    st.markdown(f'<div class="metric-card risk-high">', unsafe_allow_html=True)
                    st.markdown(f"### {risk_icon} {risk_level}")
                    st.markdown(f"**Churn Probability:** {churn_probability:.1%}")
                    st.markdown("**Status:** Customer is likely to **CHURN**")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card risk-low">', unsafe_allow_html=True)
                    st.markdown(f"### {risk_icon} {risk_level}")
                    st.markdown(f"**Retention Probability:** {(1-churn_probability):.1%}")
                    st.markdown("**Status:** Customer is likely to **STAY**")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Risk gauge
                st.plotly_chart(create_risk_gauge(churn_probability), use_container_width=True)

                # Recommendations
                st.subheader("💡 Actionable Recommendations")
                recommendations = generate_recommendations(prediction[0], churn_probability, summary_df)

                for rec in recommendations:
                    st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please check your input values and try again.")

with tab2:
    st.markdown("### 📊 Analytics Dashboard")

    # Key Performance Indicators with enhanced metrics
    st.subheader("Key Performance Indicators")

    # Calculate dynamic metrics
    credit_score = summary_df['CreditScore'][0]
    balance = summary_df['Balance'][0]
    age = summary_df['Age'][0]
    products = summary_df['NumOfProducts'][0]
    tenure = summary_df['Tenure'][0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score_rating = "Excellent" if credit_score >= 750 else "Good" if credit_score >= 650 else "Fair"
        st.metric("Credit Standing", score_rating, f"{credit_score}/850")

    with col2:
        balance_tier = "VIP" if balance > 150000 else "Premium" if balance > 100000 else "Standard" if balance > 25000 else "Basic"
        st.metric("Account Tier", balance_tier, f"${balance:,.0f}")

    with col3:
        engagement_level = "High" if summary_df['Engagement Score'][0] > 7 else "Medium" if summary_df['Engagement Score'][0] > 4 else "Low"
        st.metric("Engagement", engagement_level, f"{summary_df['Engagement Score'][0]}/10")

    with col4:
        loyalty_score = min(100, (tenure * 10) + (products * 15) + (1 if summary_df['IsActiveMember'][0] else 0) * 20)
        st.metric("Loyalty Score", f"{loyalty_score}%", f"{tenure} yrs tenure")

    # Customer Profile Radar Chart
    st.subheader("Customer Profile Analysis")
    radar_chart = create_customer_profile_chart(summary_df)
    st.plotly_chart(radar_chart, use_container_width=True)

    # Historical comparison (simulated data)
    st.subheader("📈 Historical Trends Comparison")

    # Generate sample historical data for comparison
    historical_data = {
        'Age_Group': ['18-25', '26-35', '36-45', '46-55', '56+'],
        'Avg_Churn_Rate': [25, 18, 15, 12, 8],
        'Current_Profile': [age] * 5
    }

    current_age_group = '18-25' if age <= 25 else '26-35' if age <= 35 else '36-45' if age <= 45 else '46-55' if age <= 55 else '56+'
    current_churn_rate = next((rate for group, rate in zip(historical_data['Age_Group'], historical_data['Avg_Churn_Rate']) if group == current_age_group), 15)

    col1, col2 = st.columns(2)

    with col1:
        # Churn rate comparison
        fig = px.bar(
            x=historical_data['Age_Group'],
            y=historical_data['Avg_Churn_Rate'],
            title="Churn Rate by Age Group",
            labels={'x': 'Age Group', 'y': 'Churn Rate (%)'},
            color=historical_data['Age_Group'],
            color_discrete_map={current_age_group: '#1f77b4'}
        )
        fig.add_hline(y=current_churn_rate, line_dash="dash", line_color="red",
                     annotation_text=f"Your group's avg: {current_churn_rate}%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk factors analysis
        risk_factors = {
            'Factor': ['Low Credit Score', 'Low Balance', 'Few Products', 'Low Engagement', 'Short Tenure'],
            'Impact': [0.3, 0.25, 0.2, 0.15, 0.1],
            'Your_Status': [
                'High Risk' if credit_score < 600 else 'Medium Risk' if credit_score < 700 else 'Low Risk',
                'High Risk' if balance < 25000 else 'Medium Risk' if balance < 75000 else 'Low Risk',
                'High Risk' if products == 1 else 'Medium Risk' if products == 2 else 'Low Risk',
                'High Risk' if summary_df['Engagement Score'][0] < 3 else 'Medium Risk' if summary_df['Engagement Score'][0] < 7 else 'Low Risk',
                'High Risk' if tenure < 2 else 'Medium Risk' if tenure < 5 else 'Low Risk'
            ]
        }

        fig = px.scatter(
            risk_factors,
            x='Factor',
            y='Impact',
            size='Impact',
            color='Your_Status',
            title="Risk Factor Analysis",
            color_discrete_map={'Low Risk': '#388e3c', 'Medium Risk': '#f57c00', 'High Risk': '#d32f2f'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 👥 Customer Insights & Segmentation")

    # Customer persona analysis
    st.subheader("Customer Persona Analysis")

    age = summary_df['Age'][0]
    balance = summary_df['Balance'][0]
    credit_score = summary_df['CreditScore'][0]
    geography = geo_val
    gender = gen_val

    # Determine persona
    if age < 30 and credit_score > 700:
        persona = "Young Professional"
        description = "Ambitious millennial with high credit score, likely tech-savvy and career-focused."
        behaviors = ["High digital engagement", "Multiple products usage", "Responsive to rewards", "Price sensitive"]
    elif age > 50 and balance > 100000:
        persona = "Established Senior"
        description = "Experienced customer with substantial savings, values stability and personalized service."
        behaviors = ["Prefers traditional banking", "Loyal to established relationships", "Responds to security features", "Less price sensitive"]
    elif geography == "Germany" and credit_score > 650:
        persona = "German Business Professional"
        description = "Detail-oriented professional from Germany, focuses on efficiency and long-term planning."
        behaviors = ["Values precision and reliability", "Interested in investment products", "Prefers comprehensive services"]
    elif gender == "Female" and balance > 75000:
        persona = "High-Value Female Customer"
        description = "Successful female customer who values financial security and personalized advice."
        behaviors = ["Interested in wealth management", "Responds to educational content", "Values relationship building"]
    else:
        persona = "Standard Customer"
        description = "Typical banking customer seeking reliable financial services."
        behaviors = ["Uses basic banking products", "Responds to promotions", "Values convenience"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**🎭 Persona:** {persona}")
        st.markdown(f"**📍 Location:** {geography}")
        st.markdown(f"**👤 Gender:** {gender}")

    with col2:
        st.markdown(f"**📝 Description:** {description}")

        st.markdown("**🔍 Key Behaviors:**")
        for behavior in behaviors:
            st.markdown(f"• {behavior}")

    # Churn risk by segment
    st.subheader("Segment Risk Analysis")

    segments_data = {
        'Segment': ['Young Professional', 'Established Senior', 'German Professional', 'High-Value Female', 'Standard Customer'],
        'Avg_Churn_Rate': [22, 8, 12, 15, 18],
        'Segment_Size': [25, 20, 18, 15, 22]
    }

    current_segment_idx = segments_data['Segment'].index(persona) if persona in segments_data['Segment'] else 4
    current_churn_rate = segments_data['Avg_Churn_Rate'][current_segment_idx]

    fig = px.bar(
        segments_data,
        x='Segment',
        y='Avg_Churn_Rate',
        title="Churn Rate by Customer Segment",
        color='Segment',
        color_discrete_map={persona: '#1f77b4'}
    )
    fig.add_annotation(
        x=persona,
        y=current_churn_rate,
        text=f"Your segment: {current_churn_rate}% churn rate",
        showarrow=True,
        arrowhead=1
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 📋 Batch Customer Analysis")

    st.markdown("""
    Upload a CSV file with multiple customer records to analyze churn risk in batch.
    The file should contain columns: Geography, Gender, CreditScore, Age, Tenure,
    Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, EngagementScore
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} customer records")

            # Process batch predictions
            if st.button("🔍 Analyze All Customers", type="primary"):
                with st.spinner("Analyzing customer data..."):
                    # Prepare data for prediction
                    batch_predictions = []
                    for idx, row in batch_data.iterrows():
                        try:
                            model_input = build_model_input(
                                row['Geography'],
                                row['Gender'],
                                row['CreditScore'],
                                row['Age'],
                                row['Tenure'],
                                row['Balance'],
                                row['NumOfProducts'],
                                row['HasCrCard'],
                                row['IsActiveMember'],
                                row['EstimatedSalary'],
                                row.get('EngagementScore', 5.0)  # Default if missing
                            )
                            pred_df = pd.DataFrame([model_input])
                            prediction = model.predict(pred_df)[0]
                            probability = model.predict_proba(pred_df)[0][1] if hasattr(model, 'predict_proba') else prediction

                            batch_predictions.append({
                                'Customer_ID': idx + 1,
                                'Churn_Prediction': 'High Risk' if prediction == 1 else 'Low Risk',
                                'Churn_Probability': probability,
                                'Geography': row['Geography'],
                                'Age': row['Age'],
                                'Balance': row['Balance']
                            })
                        except Exception as e:
                            batch_predictions.append({
                                'Customer_ID': idx + 1,
                                'Churn_Prediction': 'Error',
                                'Churn_Probability': 0,
                                'Geography': row['Geography'],
                                'Age': row['Age'],
                                'Balance': row['Balance']
                            })

                    results_df = pd.DataFrame(batch_predictions)

                    # Display results
                    st.subheader("Batch Analysis Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Summary statistics
                    high_risk_count = len(results_df[results_df['Churn_Prediction'] == 'High Risk'])
                    total_count = len(results_df)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", total_count)
                    with col2:
                        st.metric("High Risk Customers", high_risk_count)
                    with col3:
                        st.metric("Churn Rate", f"{(high_risk_count/total_count*100):.1f}%")

                    # Export results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="churn_analysis_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct column names and data types.")

# --- 4. Visual Analysis Section ---
st.markdown("---")
st.header("📊 Model Analysis & Insights")

# Using columns for the first two charts
col_img1, col_img2 = st.columns(2)

with col_img1:
    st.subheader("Feature Importance")
    # Matching your filename: shap.png
    st.image("./shap.png", caption="Impact of features on prediction", use_container_width=True)

with col_img2:
    st.subheader("Model Performance Metrics")
    # Matching your filename: download (1).png
    st.image("./download (1).png", caption="Confusion Matrix / Accuracy Metrics", use_container_width=True)

# Optional: SHAP explanation in an expander for a professional touch
with st.expander("ℹ️ What do these charts mean?"):
    st.write("""
        The **SHAP** chart shows which factors (like Age, Balance, or Number of Products)
        most influenced the model's decision to predict Churn or Stay.
        The **Performance** chart shows how accurate the model was during testing.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏦 Bank Customer Churn Intelligence Pro | Powered by Machine Learning</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)