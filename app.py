import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Oil & SRP Analytics", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🛢️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Production Data (CSV)", type="csv")

# --- MAIN PAGE ---
st.title("Advanced Oil Well & SRP Analytics Suite")

tab1, tab2 = st.tabs(["📈 Production Analytics (Model A)", "⚙️ SRP Mechanical Health (Model B)"])

with tab1:
    if uploaded_file is not None:
        try:
            # Data load karne
            df = pd.read_csv(uploaded_file)
            
            # Date column conversion
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            st.subheader("Real-time Production Overview")
            
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Oil Production", f"{df['oil'].mean():.2f} bbl")
            col2.metric("Avg Downhole Pressure", f"{df['down_hole_presure'].mean():.2f} psi")
            col3.metric("Total Maintenance Events", int(df['maintenance_event'].sum()))
            col4.metric("Avg Wellhead Temp", f"{df['well_head_temperature'].mean():.2f} °C")

            st.divider()
            
            # 1. Production vs Pressure Trend
            st.write("### Oil Production vs Downhole Pressure Trend")
            fig = px.line(df, x='date' if 'date' in df.columns else df.index, 
                         y=['oil', 'down_hole_presure'], 
                         title="Production vs Pressure Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # 2. Correlation Heatmap
            st.write("### Feature Correlation")
            relevant_cols = ['oil', 'down_hole_presure', 'well_head_presure', 'choke_size_pct']
            # Fakt aslele columns check karne
            available_cols = [c for c in relevant_cols if c in df.columns]
            corr = df[available_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}. Please check your CSV columns.")
            
    else:
        st.info("Please upload the Production CSV file from the sidebar to start.")

with tab2:
    st.subheader("SRP Mechanical Monitoring")
    st.warning("Upload SRP technical data to activate this module.")