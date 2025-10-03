import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import lightkurve as lk
import warnings
from sklearn.model_selection import train_test_split
import google.generativeai as genai
import os
import shutil

# --- Page Configuration ---
st.set_page_config(page_title="Exoplanet Discovery Engine", page_icon="🚀", layout="wide")
warnings.filterwarnings('ignore', category=UserWarning)

# --- Initialize Session State for Logbook ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- API Key & Model Loading ---
# (Same as before)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.warning("⚠️ Gemini API key not found.")
    genai = None
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/exoplanet_model_final.h5')
        return model
    except Exception: return None
model = load_model()

# --- LLM Function ---
# (Same as before)
@st.cache_data
def get_llm_explanation(_data_dict):
    if not genai: return "AI Science Communicator is disabled."
    gen_model = genai.GenerativeModel('gemini-pro-latest')  # Use the latest Gemini Pro model
    prompt = f"You are an expert astronomer for NASA. Explain the results of an exoplanet transit search for '{_data_dict['star_name']}' in an exciting, one-paragraph summary. Data: Orbital Period: {_data_dict['period']:.4f} days, Transit Duration: {_data_dict['duration']:.2f} hours, Transit Depth: {_data_dict['depth']:.4f}. Explain these simply (e.g., period is the planet's 'year')."
    try:
        return gen_model.generate_content(prompt).text
    except Exception: return "Could not generate explanation."

# --- Main App Interface ---
st.title('🚀 AI Exoplanet Discovery Engine')
st.write("A tool to automatically discover and analyze exoplanet candidates from live NASA data, powered by AI-driven principles.")
tab1, tab2, tab3 = st.tabs(["🛰️ Live Discovery Engine", "🔬 Manual Analysis", "📖 Discovery Logbook"])

# ==============================================================================
# TAB 1: Live Discovery Engine
# ==============================================================================
with tab1:
    st.header("Step 1: Select a Star for Analysis")
    mission = st.selectbox("Select Mission:", ("Kepler", "TESS"))
    if mission == "Kepler": id_label, default_id, id_prefix = "Kepler ID (KIC)", "6541920", "KIC"
    else: id_label, default_id, id_prefix = "TESS ID (TIC)", "150428135", "TIC"
    target_id = st.text_input(f"Enter {id_label}", default_id)

    if st.button("Begin Discovery!", key="fetch_button", type="primary"):
        if not target_id: st.warning(f"Please enter a {id_label}.")
        else:
            try:
                search_string = f"{id_prefix} {target_id}"
                with st.spinner(f"Contacting NASA's MAST Archive for {search_string}..."):
                    search_result = lk.search_lightcurve(search_string, mission=mission)
                    if not search_result: st.error(f"No data found for this ID in {mission}."); st.stop()
                with st.spinner(f"Downloading light curve data..."):
                    lc = search_result[0:5].download_all().stitch().remove_nans().normalize().remove_outliers()
                st.header("Step 2: Automated Transit Detection")
                st.info("Our engine is now running an algorithm (Box Least Squares) to find the most probable periodic transit signal.")
                with st.spinner("Analyzing light curve for hidden signals..."):
                    bls = lc.to_periodogram(method='bls')
                    period = bls.period_at_max_power
                    duration = bls.duration_at_max_power
                    depth_value_raw = bls.depth_at_max_power.value
                    depth_value = depth_value_raw if np.isscalar(depth_value_raw) else depth_value_raw[0]
                    in_transit_mask = bls.get_transit_mask(period=period, transit_time=bls.transit_time_at_max_power, duration=duration)
                st.success("Analysis Complete! A high-potential transit signal was detected.")
                st.header("Step 3: Discovery Results & AI Interpretation")
                c1, c2, c3 = st.columns(3)
                c1.metric("Orbital Period (days)", f"{period.value:.4f}")
                c2.metric("Transit Duration (hours)", f"{(duration.to('h')).value:.2f}")
                c3.metric("Transit Depth", f"{depth_value:.4f}")
                
                # --- ADD TO LOGBOOK ---
                log_entry = {
                    "Star Name": lc.label,
                    "Period (days)": f"{period.value:.4f}",
                    "Duration (hours)": f"{(duration.to('h')).value:.2f}",
                    "Depth": f"{depth_value:.4f}"
                }
                st.session_state.history.append(log_entry)
                # ----------------------

                st.subheader("🤖 AI Science Communicator's Report")
                explanation_data = {"star_name": lc.label, "period": period.value, "duration": (duration.to('h')).value, "depth": depth_value}
                explanation = get_llm_explanation(explanation_data)
                st.write(explanation)
                st.subheader("Visual Proof of Discovery")
                plot_df = lc.to_pandas().reset_index(); plot_df['highlight'] = np.where(in_transit_mask, 'In Transit', 'Out of Transit')
                fig = px.scatter(plot_df, x='time', y='flux', color='highlight', color_discrete_map={'In Transit': 'red', 'Out of Transit': 'blue'}, title=f'Interactive Light Curve for {lc.label}')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"An error occurred: {e}")

# ==============================================================================
# TAB 2: Manual Analysis
# ==============================================================================
with tab2:
    # (The code for Tab 2 remains the same as before)
    st.title("🔬 Manual Data Analysis")
    st.write("Upload your own custom data files to classify and analyze a star system.")
    if model is None: st.error("AI Classifier model is not loaded.")
    else:
        features_file = st.file_uploader("Upload Features CSV", type="csv")
        lightcurve_file = st.file_uploader("Upload Light Curve CSV", type="csv")
        if features_file and lightcurve_file:
            if st.button("Analyze Uploaded Data", type="primary"):
                # ... (rest of the manual analysis code)
                st.header("Step 2: AI Classifier Result")
                features_df = pd.read_csv(features_file)
                features_values = features_df.values
                features_reshaped = features_values.reshape(1, -1, 1)
                prediction_prob = model.predict(features_reshaped, verbose=0)[0][0]
                st.metric("AI Model's Prediction", "Planet Candidate" if prediction_prob > 0.5 else "Not a Planet")
                st.metric("AI Confidence Score", f"{prediction_prob:.2%}")
                st.header("Step 3: Discovery Engine Analysis")
                lc_df = pd.read_csv(lightcurve_file)
                lc = lk.LightCurve(time=lc_df['time'], flux=lc_df['flux']).remove_nans().normalize().remove_outliers()
                bls = lc.to_periodogram(method='bls')
                period = bls.period_at_max_power
                duration = bls.duration_at_max_power
                depth_value_raw = bls.depth_at_max_power.value
                depth_value = depth_value_raw if np.isscalar(depth_value_raw) else depth_value_raw[0]
                in_transit_mask = bls.get_transit_mask(period=period, transit_time=bls.transit_time_at_max_power, duration=duration)
                st.subheader("Detected Signal Properties")
                c1, c2, c3 = st.columns(3); c1.metric("Orbital Period (days)", f"{period.value:.4f}"); c2.metric("Transit Duration (hours)", f"{(duration.to('h')).value:.2f}"); c3.metric("Transit Depth", f"{depth_value:.4f}")
                st.subheader("Visual Proof")
                plot_df = lc.to_pandas().reset_index(); plot_df['highlight'] = np.where(in_transit_mask, 'In Transit', 'Out of Transit')
                fig = px.scatter(plot_df, x='time', y='flux', color='highlight', color_discrete_map={'In Transit': 'red', 'Out of Transit': 'blue'})
                st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3: Discovery Logbook - NEW
# ==============================================================================
with tab3:
    st.header("📖 Your Discovery Logbook")
    st.write("This logbook keeps a temporary record of the potential candidates you've discovered during this session.")
    
    if not st.session_state.history:
        st.info("You haven't discovered any candidates yet in this session. Go to the 'Live Discovery Engine' to find some!")
    else:
        # Convert the list of dictionaries to a pandas DataFrame for nice display
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear Logbook"):
            st.session_state.history = []
            st.rerun()