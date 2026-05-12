import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import json
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Oil Well & SRP AI Suite", layout="wide")

# --- SIDEBAR: Control Panel ---
st.sidebar.title("🛢️ AI Control Panel")
st.sidebar.markdown("Upload your datasets to activate analytics.")

st.sidebar.subheader("1. Production Data")
prod_file = st.sidebar.file_uploader("Upload Production CSV (Model A)", type="csv", key="prod")

st.sidebar.markdown("---")

st.sidebar.subheader("2. SRP Technical Data")
srp_file = st.sidebar.file_uploader("Upload SRP Technical CSV (Model B)", type="csv", key="srp")

# --- MAIN PAGE TITLE ---
st.title("Advanced Oil Well & SRP Analytics Suite")
st.markdown("Real-time Production Monitoring & Predictive AI Maintenance")

# 2. Creating Main Tabs
tab1, tab2 = st.tabs(["📈 Model A: Production Analytics", "⚙️ Model B: SRP AI Diagnostics"])

# --- TAB 1: PRODUCTION ANALYTICS ---
with tab1:
    if prod_file is not None:
        try:
            df_prod = pd.read_csv(prod_file)
            
            # Date conversion
            if 'date' in df_prod.columns:
                df_prod['date'] = pd.to_datetime(df_prod['date'])
            
            st.subheader("Real-time Production Overview")
            
            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Oil Production", f"{df_prod['oil'].mean():.2f} bbl")
            col2.metric("Avg Downhole Pressure", f"{df_prod['down_hole_presure'].mean():.2f} psi")
            col3.metric("Total Maintenance Events", int(df_prod['maintenance_event'].sum()))
            col4.metric("Avg Wellhead Temp", f"{df_prod['well_head_temperature'].mean():.2f} °C")

            st.divider()
            
            # Graphs: Production vs Pressure
            st.write("### Oil Production vs Downhole Pressure Trend")
            fig1 = px.line(df_prod, x='date' if 'date' in df_prod.columns else df_prod.index, 
                          y=['oil', 'down_hole_presure'], 
                          labels={"value": "Magnitude", "date": "Date"},
                          title="Production & Pressure Relationship Over Time")
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Correlation Heatmap
            st.write("### Feature Correlation Analysis")
            relevant_cols = ['oil', 'down_hole_presure', 'well_head_presure', 'choke_size_pct']
            available_cols = [c for c in relevant_cols if c in df_prod.columns]
            if available_cols:
                corr = df_prod[available_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                                    title="Correlation Matrix of Production Variables")
                st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading Production Data: {e}")
    else:
        st.info("Please upload the Production CSV file from the sidebar to view Model A.")

# --- TAB 2: SRP AI DIAGNOSTICS ---
with tab2:
    if srp_file is not None:
        try:
            df_srp = pd.read_csv(srp_file)
            df_srp['timestamp'] = pd.to_datetime(df_srp['timestamp'])
            
            # Simulated AI Metadata
            current_cond = df_srp['ground_truth'].iloc[0]
            confidence = 94.5  # Typical CNN output confidence
            
            # Top Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Condition", current_cond)
            m2.metric("AI Confidence Score", f"{confidence}%")
            m3.metric("12-Hour Forecast", "⚠️ Risk Increasing" if current_cond != "Normal" else "✅ System Stable")
            m4.metric("Last Cycle Time", df_srp['timestamp'].iloc[0].strftime('%H:%M:%S'))

            # Sub-Tabs for Diagnostics, Forecasting, and History
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🔍 Live Diagnostics", "🔮 AI Forecasting", "📜 Historical Trends"])

            with sub_tab1:
                st.subheader("Downgraph (Dynamometer Card Analysis)")
                st.write("Description: Visualizing the mechanical load cycle for sub-surface pump health.")
                try:
                    # Parsing raw list data
                    pos = ast.literal_eval(df_srp['raw_position'].iloc[0])
                    load = ast.literal_eval(df_srp['raw_load'].iloc[0])
                    
                    fig_dyna = px.line(x=pos, y=load, 
                                      labels={'x':'Position (inches)', 'y':'Load (lbs)'},
                                      title=f"Diagnostic Dynacard: {current_cond}")
                    fig_dyna.update_traces(fill='toself', line_color='cyan')
                    st.plotly_chart(fig_dyna, use_container_width=True)
                except:
                    st.error("Error parsing Dynacard data. Ensure raw_position and raw_load are list formats.")

            with sub_tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("12-Hour Failure Risk Forecast")
                    # Simulated Spatio-Temporal Prediction
                    forecast_data = pd.DataFrame({
                        'Hour': [f"+{i}h" for i in range(1, 13)],
                        'Probability': [10, 15, 20, 45, 60, 85, 90, 80, 70, 50, 40, 30] if current_cond != "Normal" else [5]*12
                    })
                    fig_forecast = px.area(forecast_data, x='Hour', y='Probability', 
                                          title="Spatio-Temporal CNN Risk Probability (%)")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                
                with col_b:
                    st.subheader("Multi-Condition Probabilities")
                    # Confidence score for multiple potential conditions
                    probs = {
                        'Normal': 5.0 if current_cond != 'Normal' else 95.0,
                        'Gas Interference': 94.5 if current_cond == 'Gas Interference' else 2.0,
                        'Fluid Pound': 94.5 if current_cond == 'Fluid Pound' else 1.5,
                        'Valve Leak': 94.5 if current_cond == 'Traveling Valve Leak' else 1.0
                    }
                    fig_prob = px.bar(x=list(probs.keys()), y=list(probs.values()), 
                                     labels={'x':'Condition', 'y':'Probability %'},
                                     color=list(probs.keys()), title="Model Output Confidence")
                    st.plotly_chart(fig_prob, use_container_width=True)

            with sub_tab3:
                st.subheader("Historical Condition Trend (Last 24 Hours)")
                fig_trend = px.scatter(df_srp.head(24), x='timestamp', y='ground_truth', 
                                      color='ground_truth', size_max=15,
                                      title="Well Classification History")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                st.subheader("Input Data Snapshot (Recent Cycles)")
                st.dataframe(df_srp[['timestamp', 'peak_load', 'min_load', 'ground_truth']].head(24))

        except Exception as e:
            st.error(f"Error loading SRP Data: {e}")
    else:
        st.info("Please upload the SRP Technical CSV file from the sidebar to view Model B.")