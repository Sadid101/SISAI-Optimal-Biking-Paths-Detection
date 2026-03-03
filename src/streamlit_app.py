import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
import io
import json

warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import functions from app.py
from utils.pedestrianDataFetcher import getAndPrintListOfSites, check_files_exist, fetchEcoCounterSiteData, storeSplitJson
from utils.weather_dataset_builder import check_if_all_historical_weather_data_exists, run_builder
from utils.forecast_fetcher import fetchAndSaveForecast
from app import (
    load_weather, 
    load_pedestrians_training_data,
    load_pedestrians_testing_data,
    load_weather as load_forecast,
    add_time_features,
    processDataWithRandomForestRegression,
    processDataWithLinearRegression,
    LOCAL_TZ,
    WEATHER_FORECAST_JSON_PATH
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🚶 Pedestrian Traffic Predictor",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1em;
        color: #555;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & CACHING
# ============================================================================
def load_sites_list():
    """Load and cache the list of all sites with multiple fallbacks"""
    
    # Check if already cached in session state
    if 'sites_list' in st.session_state:
        return st.session_state.sites_list
    
    # Strategy 1: Try to load from local JSON first (fastest, most reliable)
    try:
        json_path = Path(__file__).parent / "data" / "pedestrians" / "ecoCounterSites.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sites = data.get('ecoCounterSites', [])
            if sites and len(sites) > 0:
                # Validate all sites have required fields
                valid_sites = [s for s in sites if s and isinstance(s, dict) and s.get('siteId')]
                if valid_sites:
                    st.session_state.sites_list = valid_sites
                    return valid_sites
    except Exception as e:
        pass  # Continue to next strategy
    
    # Strategy 2: Try API call with output suppression
    try:
        # Suppress console output from API call
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sites = getAndPrintListOfSites()
        finally:
            sys.stdout = old_stdout
        
        if sites and len(sites) > 0:
            # Validate all sites
            valid_sites = [s for s in sites if s and isinstance(s, dict) and s.get('siteId')]
            if valid_sites:
                st.session_state.sites_list = valid_sites
                return valid_sites
    except Exception as e:
        pass  # Continue to next strategy
    
    # Strategy 3: Return empty list
    return []

def get_site_options():
    """Get formatted site options for dropdown"""
    sites = load_sites_list()
    
    # Validate sites list
    if not sites or len(sites) == 0:
        return {"❌ No sites available": {"siteId": None, "name": "N/A", "domain": "N/A"}}
    
    # Filter out None entries and create options
    valid_sites = {}
    for s in sites:
        if s is None:
            continue
        try:
            if s.get('name') and s.get('siteId'):
                key = f"📍 {s['name']} ({s['siteId']})"
                valid_sites[key] = s
        except (TypeError, AttributeError):
            continue
    
    if not valid_sites:
        return {"❌ No valid sites found": {"siteId": None, "name": "N/A", "domain": "N/A"}}
    
    return valid_sites

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def validate_and_fetch_pedestrian_data(site_id, domain):
    """Check if pedestrian data exists, if not fetch it"""
    if check_files_exist(site_id):
        st.success(f"✅ Using existing pedestrian data for site {site_id}")
        return True
    else:
        st.warning(f"⏳ Fetching pedestrian data for site {site_id}...")
        try:
            data = fetchEcoCounterSiteData(site_id, domain, 'hour')
            storeSplitJson(data, site_id=site_id)
            st.success(f"✅ Data fetched and saved successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error fetching pedestrian data: {e}")
            return False

def validate_and_fetch_weather_data():
    """Check if weather data exists, if not fetch it"""
    if check_if_all_historical_weather_data_exists():
        st.success("✅ Historical weather data found")
        return True
    else:
        st.warning("⏳ Fetching historical weather data...")
        try:
            run_builder()
            st.success("✅ Weather data fetched successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error fetching weather data: {e}")
            return False

def fetch_forecast_data():
    """Fetch fresh weather forecast"""
    try:
        st.info("🌤️ Updating weather forecast...")
        fetchAndSaveForecast()
        st.success("✅ Forecast updated successfully!")
        return True
    except Exception as e:
        st.warning(f"⚠️ Could not fetch fresh forecast: {e}")
        return False

def run_prediction(site, start_date, end_date, model_type):
    """Run the prediction and return results"""
    try:
        # Validate site
        if site is None or site.get('siteId') is None:
            st.error("❌ Invalid site selected")
            return None
        
        # Ensure data exists
        with st.spinner("📊 Validating data sources..."):
            try:
                validate_and_fetch_pedestrian_data(site['siteId'], site['domain'])
            except Exception as e:
                st.warning(f"⚠️ Pedestrian data fetch issue: {e}")
            
            try:
                validate_and_fetch_weather_data()
            except Exception as e:
                st.warning(f"⚠️ Weather data fetch issue: {e}")
            
            try:
                fetch_forecast_data()
            except Exception as e:
                st.warning(f"⚠️ Forecast fetch issue: {e}")
        
        # Train model
        with st.spinner("🤖 Training model (this may take a moment)..."):
            try:
                if model_type == "Random Forest":
                    model, featureColumns, coef, intercept = processDataWithRandomForestRegression(site_id=site['siteId'])
                else:
                    model, featureColumns, coef, intercept = processDataWithLinearRegression(site_id=site['siteId'])
            except Exception as e:
                st.error(f"❌ Model training failed: {e}")
                return None
        
        # Load forecasted weather
        try:
            forecast_df = load_forecast(WEATHER_FORECAST_JSON_PATH)
            if forecast_df is None or len(forecast_df) == 0:
                st.error("❌ No weather forecast data available")
                return None
        except Exception as e:
            st.error(f"❌ Failed to load weather forecast: {e}")
            return None
        
        # Generate predictions
        with st.spinner("Generating predictions..."):
            date_range = pd.date_range(start=start_date, end=end_date, freq="h", tz=LOCAL_TZ)
            predictions = []
            
            for query_ts in date_range:
                try:
                    query_ts = pd.Timestamp(query_ts).floor("h")
                    
                    # Find matching forecast temperature
                    matching_row = forecast_df[forecast_df['ts_hour'] == query_ts]
                    
                    if not matching_row.empty:
                        query_temp = matching_row.iloc[0]['temp_c']
                    else:
                        # Fallback to last known temperature
                        query_temp = forecast_df['temp_c'].iloc[-1]
                    
                    # Make prediction
                    query_df = pd.DataFrame([{"ts_hour": query_ts, "temp_c": query_temp}])
                    query_df = add_time_features(query_df, "ts_hour")
                    y_hat = model.predict(query_df[featureColumns])[0]
                    
                    predictions.append({
                        "Timestamp": query_ts,
                        "Temperature (°C)": round(query_temp, 1),
                        "Predicted Pedestrians": round(max(0, y_hat), 0)
                    })
                except Exception as e:
                    st.warning(f"⚠️ Skipped prediction for {query_ts}: {e}")
                    continue
            
            if not predictions:
                st.error("❌ No predictions were generated")
                return None
            
            prediction_df = pd.DataFrame(predictions)
            return prediction_df
    
    except Exception as e:
        st.error(f"❌ Unexpected error during prediction: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Header
st.markdown("# 🚶 Pedestrian Traffic Predictor")
st.markdown("### Oulu City - Hourly Pedestrian Count Predictions")
st.markdown("---")

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # Site Selection
    st.subheader("📍 Location Selection")
    
    try:
        with st.spinner("📡 Loading sites..."):
            site_options = get_site_options()
        
        if site_options is None or len(site_options) == 0:
            st.error("❌ Unable to load sites. Ensure internet connection and try refreshing.")
            st.stop()
        
        site_names = list(site_options.keys())
        
        # Check if we have a valid site in options
        if all('❌' in name for name in site_names):
            st.error("❌ No valid sites available. Please refresh the page.")
            st.stop()
        
        selected_site_name = st.selectbox(
            label="Search and select a location",
            options=site_names,
            help="Type to search by location name or ID"
        )
        selected_site = site_options.get(selected_site_name)
        
        # Validate site selection
        if selected_site is None or selected_site.get('siteId') is None:
            st.error("❌ Please select a valid site")
            st.stop()
        
        # Display selected site info
        with st.expander("📌 Site Details", expanded=False):
            st.write(f"**Name:** {selected_site['name']}")
            st.write(f"**Site ID:** {selected_site['siteId']}")
            st.write(f"**Domain:** {selected_site['domain']}")
            st.write(f"**Timezone:** {selected_site.get('timezone', 'N/A')}")
        
    except Exception as e:
        st.error(f"❌ Error loading sites: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Date Range Selection
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)
    
    min_date = datetime.now()
    max_date = min_date + timedelta(days=7)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=min_date + timedelta(days=1),
            min_value=start_date,
            max_value=max_date
        )
    
    # Time selection
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.time_input("Start Time", value=datetime.min.time())
    with col2:
        end_time = st.time_input("End Time", value=datetime.min.time())
    
    # Combine date and time
    start_datetime = pd.Timestamp(
        datetime.combine(start_date, start_time),
        tz=LOCAL_TZ
    ).floor("h")
    
    end_datetime = pd.Timestamp(
        datetime.combine(end_date, end_time),
        tz=LOCAL_TZ
    ).floor("h")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("ML Model Selection")
    model_type = st.radio(
        "Choose prediction model:",
        ["Random Forest", "Linear Regression"],
        help="Random Forest: More accurate but slower\nLinear Regression: Faster but less accurate"
    )
    
    st.markdown("---")
    
    # Predict Button
    predict_button = st.button(
        "Generate Predictions",
        type="primary",
        use_container_width=True
    )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if predict_button:
    # Validate inputs
    if selected_site is None or selected_site.get('siteId') is None:
        st.error("❌ Please select a valid site first")
    elif start_datetime >= end_datetime:
        st.error("❌ Start date must be before end date")
    else:
        # Run prediction
        results_df = run_prediction(
            site=selected_site,
            start_date=start_datetime,
            end_date=end_datetime,
            model_type=model_type
        )
        
        if results_df is not None and len(results_df) > 0:
            # Store in session state for reuse
            st.session_state.results_df = results_df
            st.session_state.selected_site = selected_site
            st.session_state.start_date = start_datetime
            st.session_state.end_date = end_datetime
            st.success("Predictions generated successfully!")

# Display results if available
if 'results_df' in st.session_state:
    results_df = st.session_state.results_df
    selected_site = st.session_state.selected_site
    
    # Validate results
    if results_df is None or len(results_df) == 0:
        st.error("❌ No prediction results available")
        st.stop()
    
    # ========================================================================
    # METRICS SECTION
    # ========================================================================
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pedestrians = results_df['Predicted Pedestrians'].mean()
        st.metric(
            label="Average Pedestrians",
            value=f"{avg_pedestrians:.0f}",
            delta=None
        )
    
    with col2:
        peak_idx = results_df['Predicted Pedestrians'].idxmax()
        peak_time = results_df.loc[peak_idx, 'Timestamp'].strftime('%H:%M')
        peak_count = results_df.loc[peak_idx, 'Predicted Pedestrians']
        st.metric(
            label="Peak Hour",
            value=peak_time,
            delta=f"{peak_count:.0f} pedestrians"
        )
    
    with col3:
        avg_temp = results_df['Temperature (°C)'].mean()
        st.metric(
            label="Average Temperature",
            value=f"{avg_temp:.1f}°C",
            delta=None
        )
    
    with col4:
        total_predictions = len(results_df)
        st.metric(
            label="Total Hours Predicted",
            value=total_predictions,
            delta=None
        )
    
    st.markdown("---")
    
    # ========================================================================
    # INTERACTIVE CHART SECTION
    # ========================================================================
    st.markdown("### 📈 Interactive Predictions Chart")
    
    # Create dual-axis chart with Plotly
    fig = go.Figure()
    
    # Add pedestrian count trace
    fig.add_trace(go.Scatter(
        x=results_df['Timestamp'],
        y=results_df['Predicted Pedestrians'],
        name='Pedestrian Count',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=5),
        hovertemplate='<b>%{x|%H:%M}</b><br>Pedestrians: %{y:.0f}<extra></extra>'
    ))
    
    # Add temperature trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=results_df['Timestamp'],
        y=results_df['Temperature (°C)'],
        name='Temperature',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=5),
        yaxis='y2',
        hovertemplate='<b>%{x|%H:%M}</b><br>Temperature: %{y:.1f}°C<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Pedestrian Predictions for {selected_site['name']}",
        xaxis_title="Time",
        yaxis_title="Predicted Pedestrians",
        yaxis2=dict(
            title="Temperature (°C)",
            overlaying="y",
            side="right"
        ),
        hovermode='x unified',
        height=500,
        template='plotly_white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # DATA TABLE SECTION
    # ========================================================================
    st.markdown("### 📋 Detailed Predictions Table")
    
    # Create a formatted display dataframe
    display_df = results_df.copy()
    display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Show table with filtering options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**Total Records:** {len(display_df)}")
    
    with col2:
        show_rows = st.selectbox(
            "Show rows:",
            [10, 25, 50, 100, 'All'],
            label_visibility="collapsed"
        )
    
    with col3:
        if st.button("📊 View Full Table"):
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )
    
    # Show sample
    if show_rows == 'All':
        st.dataframe(display_df, use_container_width=True, height=600)
    else:
        st.dataframe(display_df.head(show_rows), use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # EXPORT SECTION
    # ========================================================================
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv = results_df.copy()
        csv['Timestamp'] = csv['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        csv_string = csv.to_csv(index=False)
        
        st.download_button(
            label="📥 Download as CSV",
            data=csv_string,
            file_name=f"predictions_{selected_site['siteId']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # JSON Export
        json_data = results_df.copy()
        json_data['Timestamp'] = json_data['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        json_string = json_data.to_json(orient='records', indent=2)
        
        st.download_button(
            label="📥 Download as JSON",
            data=json_string,
            file_name=f"predictions_{selected_site['siteId']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Excel Export
        try:
            excel_buffer = pd.ExcelWriter('/tmp/predictions.xlsx', engine='openpyxl')
            results_df.to_excel(excel_buffer, sheet_name='Predictions', index=False)
            excel_buffer.close()
            
            with open('/tmp/predictions.xlsx', 'rb') as f:
                excel_data = f.read()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"predictions_{selected_site['siteId']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except:
            st.info("💡 Install openpyxl for Excel export: pip install openpyxl")
    
    st.markdown("---")
    
    # ========================================================================
    # ADDITIONAL INFO
    # ========================================================================
    with st.expander("❓ How to Use", expanded=False):
        st.markdown("""
        **Step 1:** Select a location from the dropdown on the left
        
        **Step 2:** Choose your date range (up to 7 days in advance)
        
        **Step 3:** Select a prediction model:
        - Random Forest: Better accuracy, slower
        - Linear Regression: Faster, simpler
        
        **Step 4:** Click "Generate Predictions"
        
        **Step 5:** View results, export data as needed
        
        ---
        
        **What this app does:**
        - Analyzes 11+ years of pedestrian traffic data
        - Considers weather conditions and time patterns
        - Predicts hourly foot traffic for the next 7 days
        - Useful for urban planning, event management, infrastructure
        """)
    
    with st.expander("ℹ️ About the Models", expanded=False):
        st.markdown("""
        **Random Forest Regressor**
        - Uses 300 decision trees
        - Captures complex non-linear patterns
        - More accurate but computationally intensive
        - Recommended for production use
        
        **Linear Regression**
        - Simple linear relationships
        - Very fast predictions
        - Good for quick estimates
        - Useful for understanding feature importance
        
        **Features used:**
        - Temperature (°C)
        - Month of year
        - Day of week
        - Weekend indicator
        - Hour sine/cosine (cyclical encoding)
        """)

else:
    # Initial welcome message
    st.info("👈 Select a location and date range, then click 'Generate Predictions' to get started!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Quick Start
        
        1. **Select Location** - Browse 47 sensor sites across Oulu
        2. **Choose Dates** - Predict up to 7 days ahead
        3. **Pick Model** - Random Forest or Linear Regression
        4. **Generate** - Click the predict button
        5. **Export** - Download results as CSV/JSON
        """)
    
    with col2:
        st.markdown("""""")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>🚶 Oulu Pedestrian Traffic Predictor | SISAI Final Project | 2026</p>
</div>
""", unsafe_allow_html=True)
