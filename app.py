import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Set page config for premium look
st.set_page_config(page_title="GoldSentinel AI", layout="wide", page_icon="🪙")

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0ed3a !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* KPI Metric Cards Glow */
    [data-testid="stMetricValue"] {
        color: #f0ed3a !important;
        font-size: 2rem !important;
        font-weight: bold;
    }
    
    /* Subtext for Metrics */
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪙 GoldSentinel AI Forecast")
st.markdown("_Predictive Analytics & Machine Learning for Precious Metals_")

@st.cache_data
def load_data():
    file_path = "data/processed/powerbi_export.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'])
            if len(df) > 0:
                return df
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return None

data = load_data()

if data is None:
    st.warning("⚠️ Exported Data not found. Please run the data extraction and modeling pipeline first.")
else:
    # Sort data by Date
    data = data.sort_values(by="Date", ascending=True)
    latest_date = data["Date"].iloc[-1]
    last_actual = data["Actual_Next_Close"].iloc[-1]
    last_pred = data["Predicted_Next_Close"].iloc[-1]
    
    # Calculate Direction Accuracy overall
    accuracy = data["Direction_Correct"].mean() * 100
    avg_error = data["Prediction_Error_Percentage"].mean()

    # KPI Layout
    st.markdown("### System Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Latest Trading Date", value=latest_date.strftime("%Y-%m-%d"))
    with col2:
        st.metric(label="Predicted Next Close (per 10g)", value=f"₹{last_pred:,.2f}", delta=f"₹{last_pred - last_actual:,.2f} diff")
    with col3:
        st.metric(label="Directional Accuracy", value=f"{accuracy:.1f}%")
    with col4:
        st.metric(label="Avg Prediction Error", value=f"{avg_error:.2f}%")

    st.markdown("---")
    
    # Time Series Chart logic
    st.markdown("### 📈 Historical vs Predicted Closing Prices")
    
    # Selector for timeframe
    max_days = len(data)
    if max_days > 1:
        min_days = min(30, max_days)
        default_days = min(365, max_days)
        days_to_show = st.slider("Select days to view:", min_value=min_days, max_value=max_days, value=default_days)
    else:
        days_to_show = max_days
        
    df_plot = data.tail(days_to_show)

    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(
        x=df_plot['Date'], y=df_plot['Actual_Next_Close'],
        mode='lines', name='Actual Next Close',
        line=dict(color='#00ffcc', width=2)
    ))
    # Predicted
    fig.add_trace(go.Scatter(
        x=df_plot['Date'], y=df_plot['Predicted_Next_Close'],
        mode='lines', name='Predicted Next Close',
        line=dict(color='#ff007f', width=2, dash='dot')
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#30363d'),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance and Raw Data Table Side by Side
    col_feat, col_data = st.columns([1, 1])
    
    with col_feat:
        st.markdown("### 📊 Model Drivers (Feature Importance)")
        fi_path = "models/feature_importance.csv"
        if os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path).head(10)
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation='h',
                color="Importance", color_continuous_scale="Viridis"
            )
            fig_fi.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9')
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data missing.")
            
    with col_data:
        st.markdown("### 🗄️ Recent Data Output")
        st.dataframe(df_plot.tail(15)[['Date', 'Actual_Close', 'Predicted_Next_Close', 'Prediction_Error_Percentage']], use_container_width=True)
