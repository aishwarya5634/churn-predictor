import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import matplotlib.pyplot as plt
import sys
import os

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.retention_engine import (
    apply_retention_strategy,
    get_strategy,
    calculate_clv,
    INTERVENTION_COSTS,
    RETENTION_LIFT
)

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title='Churn Intelligence Dashboard',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─────────────────────────────────────
# LOAD MODELS & DATA
# ─────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load(os.path.join(ROOT, 'src', 'best_model.pkl'))
    scaler = joblib.load(os.path.join(ROOT, 'src', 'scaler.pkl'))
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(ROOT, 'data', 'bank_churn_with_predictions.csv'))

model, scaler = load_models()
df            = load_data()

# Define feature columns
feature_cols = [c for c in df.columns
                if c not in ['Exited', 'RowNumber',
                            'CustomerId', 'Surname',
                            'clv', 'strategy',
                            'intervention_cost',
                            'retention_lift',
                            'expected_value',
                            'priority',
                            'churn_prob']]

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
st.sidebar.title('🏦 Churn Intelligence')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigate', [
    '📊 Executive Summary',
    '🔍 Customer Risk Table',
    '👤 Customer Explainer',
    '🧪 A/B Test Results'
])
st.sidebar.markdown('---')
st.sidebar.markdown(f'**Total Customers:** {len(df):,}')
st.sidebar.markdown(f'**High Risk:** {(df["churn_prob"] > 0.6).sum():,}')
st.sidebar.markdown(f'**Avg Churn Prob:** {df["churn_prob"].mean():.1%}')

# ─────────────────────────────────────
# PAGE 1 — EXECUTIVE SUMMARY
# ─────────────────────────────────────
if page == '📊 Executive Summary':
    st.title('📊 Executive Summary')
    st.subheader('Customer Churn Risk Overview')

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total     = len(df)
    high_risk = (df['churn_prob'] > 0.6).sum()
    avg_prob  = df['churn_prob'].mean()
    clv_at_risk = df[df['churn_prob'] > 0.5]['clv'].sum()

    col1.metric('Total Customers',    f'{total:,}')
    col2.metric('High Risk (>60%)',   f'{high_risk:,}',
                delta=f'{high_risk/total:.1%} of base')
    col3.metric('Avg Churn Prob',     f'{avg_prob:.1%}')
    col4.metric('CLV at Risk',        f'${clv_at_risk:,.0f}')

    st.markdown('---')

    # Row 2 charts
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x='churn_prob', nbins=50,
            color_discrete_sequence=['#00C2FF'],
            title='Distribution of Churn Probabilities',
            labels={'churn_prob': 'Churn Probability'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        strategy_counts = df['strategy'].value_counts().reset_index()
        strategy_counts.columns = ['Strategy', 'Count']
        fig2 = px.bar(
            strategy_counts,
            x='Strategy', y='Count',
            title='Recommended Strategies Distribution',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 3 charts
    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.scatter(
            df, x='clv', y='churn_prob',
            color='priority',
            title='CLV vs Churn Probability',
            labels={
                'clv': 'Customer Lifetime Value ($)',
                'churn_prob': 'Churn Probability'
            },
            color_discrete_map={
                'High':   'red',
                'Medium': 'orange',
                'Low':    'green'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        priority_counts = df['priority'].value_counts().reset_index()
        priority_counts.columns = ['Priority', 'Count']
        fig4 = px.pie(
            priority_counts,
            values='Count', names='Priority',
            title='Customer Priority Distribution',
            color='Priority',
            color_discrete_map={
                'High':   'red',
                'Medium': 'orange',
                'Low':    'green'
            }
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Business Impact
    st.markdown('---')
    st.subheader('💰 Estimated Business Impact')
    col1, col2, col3 = st.columns(3)
    col1.metric('Net Annual Benefit',    '$43,972,183')
    col2.metric('ROI on Interventions',  '4,847%')
    col3.metric('Churn Reduction',       '33.9%')

# ─────────────────────────────────────
# PAGE 2 — CUSTOMER RISK TABLE
# ─────────────────────────────────────
elif page == '🔍 Customer Risk Table':
    st.title('🔍 Customer Risk Table')

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_prob = st.slider('Minimum Churn Probability', 0.0, 1.0, 0.5)
    with col2:
        priorities = st.multiselect(
            'Priority',
            ['High', 'Medium', 'Low'],
            default=['High', 'Medium']
        )
    with col3:
        strategies = st.multiselect(
            'Strategy',
            df['strategy'].unique().tolist(),
            default=df['strategy'].unique().tolist()
        )

    # Filter data
    filtered = df[
        (df['churn_prob'] >= min_prob) &
        (df['priority'].isin(priorities)) &
        (df['strategy'].isin(strategies))
    ][['Age', 'Balance', 'NumOfProducts',
    'IsActiveMember', 'churn_prob',
    'clv', 'strategy', 'priority',
    'expected_value']].copy()

    filtered = filtered.sort_values('expected_value', ascending=False)

    st.markdown(f'**Showing {len(filtered):,} customers**')

    # Display table
    st.dataframe(
        filtered.style
        .background_gradient(subset=['churn_prob'], cmap='Reds')
        .background_gradient(subset=['expected_value'], cmap='Greens')
        .format({
            'churn_prob':     '{:.1%}',
            'clv':            '${:,.0f}',
            'expected_value': '${:,.0f}',
            'Balance':        '${:,.0f}'
        }),
        use_container_width=True,
        height=500
    )

    # Download button
    st.download_button(
        '⬇️ Download as CSV',
        data=filtered.to_csv(index=False),
        file_name='at_risk_customers.csv',
        mime='text/csv'
    )

    # Summary metrics below table
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    col1.metric('Filtered Customers',  f'{len(filtered):,}')
    col2.metric('Total CLV at Risk',   f'${filtered["clv"].sum():,.0f}')
    col3.metric('Avg Churn Prob',      f'{filtered["churn_prob"].mean():.1%}')

# ─────────────────────────────────────
# PAGE 3 — CUSTOMER EXPLAINER
# ─────────────────────────────────────
elif page == '👤 Customer Explainer':
    st.title('👤 Individual Customer Explanation')

    # Customer selector
    customer_idx = st.slider(
        'Select Customer Index',
        0, len(df) - 1, 0
    )

    customer = df.iloc[customer_idx]

    # Customer KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Churn Probability', f'{customer["churn_prob"]:.1%}')
    col2.metric('CLV',               f'${customer["clv"]:,.0f}')
    col3.metric('Priority',          customer['priority'])
    col4.metric('Age',               int(customer['Age']))

    # Strategy recommendation
    risk_color = (
        '🔴' if customer['churn_prob'] > 0.6 else
        '🟡' if customer['churn_prob'] > 0.3 else
        '🟢'
    )
    st.info(f'{risk_color} **Recommended Action:** {customer["strategy"]}')

    # Customer details
    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Customer Profile')
        st.write(f'**Balance:** ${customer["Balance"]:,.0f}')
        st.write(f'**Products:** {int(customer["NumOfProducts"])}')
        st.write(f'**Tenure:** {int(customer["Tenure"])} years')
        st.write(f'**Active Member:** {"Yes" if customer["IsActiveMember"] else "No"}')
        st.write(f'**Credit Score:** {int(customer["CreditScore"])}')

    with col2:
        st.subheader('Risk Factors')
        # Generate SHAP values for this customer
        X_customer = df[feature_cols].iloc[[customer_idx]]
        X_scaled   = scaler.transform(X_customer)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_scaled_df)

        explanation = shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            data          = X_scaled_df.iloc[0],
            feature_names = feature_cols
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────
# PAGE 4 — A/B TEST RESULTS
# ─────────────────────────────────────
elif page == '🧪 A/B Test Results':
    st.title('🧪 A/B Test Results')

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Control Churn Rate',   '75.4%')
    col2.metric('Treatment Churn Rate', '49.9%',   delta='-25.5%')
    col3.metric('Churn Reduction',      '33.9%',   delta='significant')
    col4.metric('P-value',              '0.0000',  delta='✅ proven')

    st.markdown('---')

    # ROI metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Net Benefit',          '$3,664,349')
    col2.metric('Annualized Saving',    '$43,972,183')
    col3.metric('ROI',                  '4,847%')

    st.markdown('---')

    # Strategy performance chart
    strategy_data = pd.DataFrame({
        'Strategy': [
            'Dedicated Manager',
            'Email + Fee Waiver',
            'Loyalty Points',
            'Priority Call',
            'Newsletter'
        ],
        'ROI %': [7918.5, 10416.2, 16087.6, 7621.3, 123574.8],
        'Customers': [119, 236, 43, 718, 141],
        'CLV Saved': [1145048, 620454, 208820, 4157940, 348762]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            strategy_data,
            x='Strategy', y='CLV Saved',
            title='CLV Saved by Strategy ($)',
            color='CLV Saved',
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            strategy_data,
            x='Customers', y='ROI %',
            size='CLV Saved',
            text='Strategy',
            title='Strategy ROI vs Customers Targeted',
            color='ROI %',
            color_continuous_scale='Blues'
        )
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)

    # Executive summary
    st.markdown('---')
    st.subheader('📋 Executive Summary')
    st.success('''
    ✅ A/B test ran on 2,491 at-risk customers
    ✅ Treatment group received targeted retention strategies
    ✅ Churn reduced from 75.4% → 49.9% (33.9% reduction)
    ✅ Net benefit: $3,664,349 this period
    ✅ Projected annual saving: $43,972,183
    ✅ ROI: 4,847% on intervention budget
    ✅ Result is statistically significant (p=0.0000)
    ''')