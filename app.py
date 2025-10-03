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

# --- API Key Configuration & Model Loading ---
# (Same as the previous version)
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
    except Exception:
        return None

model = load_model()

# --- LLM Function ---
# (Same as the previous version)
@st.cache_data
def get_llm_explanation(_data_dict):
    if not genai:
        return "AI Science Communicator is disabled (API key not configured)."
    
    gen_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are an expert astronomer for the NASA Space Apps Challenge.
    Explain the results of an exoplanet transit search for a star named '{_data_dict['star_name']}' in an exciting, one-paragraph summary.

    Data discovered:
    - Most Probable Orbital Period: {_data_dict['period']:.4f} days
    - Estimated Transit Duration: {_data_dict['duration']:.2f} hours
    - Measured Transit Depth: {_data_dict['depth']:.4f}

    Explain these numbers simply (e.g., orbital period is the planet's "year", depth relates to its size).
    """
    try:
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception:
        return "Could not generate explanation at this time."

# ==============================================================================
# --- Main App with Sidebar Navigation ---
# ==============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Discovery Engine", "Manual Data Analysis"])

if page == "Live Discovery Engine":
    st.title('🚀 AI Exoplanet Discovery Engine')
    st.write("A unified tool to automatically discover and analyze exoplanet candidates from live NASA data, powered by AI-driven principles.")
    
    # --- Live Discovery Workflow ---
    # (Same as the previous final version)
    st.header("Step 1: Select a Star for Analysis")
    st.write("Choose a mission and enter a star's ID to fetch its data directly from NASA's archives.")
    mission = st.selectbox("Select Mission:", ("Kepler", "TESS"))
    if mission == "Kepler":
        id_label, default_id, id_prefix = "Kepler ID (KIC)", "6541920", "KIC"
    else:
        id_label, default_id, id_prefix = "TESS ID (TIC)", "150428135", "TIC"
    target_id = st.text_input(f"Enter {id_label}", default_id)

    if st.button("Begin Discovery!", key="fetch_button", type="primary"):
        # ... (The rest of the Live Discovery code is the same as before)
        if not target_id: st.warning(f"Please enter a {id_label}.")
        else:
            try:
                search_string = f"{id_prefix} {target_id}"
                with st.spinner(f"Contacting NASA's MAST Archive for {search_string}..."):
                    search_result = lk.search_lightcurve(search_string, mission=mission)
                    if not search_result: st.error(f"No data found for this ID in {mission}."); st.stop()
                with st.spinner(f"Downloading a subset of the star's light curve data..."):
                    lc_collection = search_result[0:5].download_all()
                    lc = lc_collection.stitch().remove_nans().normalize().remove_outliers()
                st.header("Step 2: Automated Transit Detection")
                st.info("Our engine is now running an advanced algorithm (Box Least Squares) to find the most probable periodic transit signal in the data.")
                with st.spinner("Analyzing light curve for hidden signals..."):
                    bls = lc.to_periodogram(method='bls')
                    period = bls.period_at_max_power; transit_time = bls.transit_time_at_max_power
                    duration = bls.duration_at_max_power
                    depth_value_raw = bls.depth_at_max_power.value
                    depth_value = depth_value_raw if np.isscalar(depth_value_raw) else depth_value_raw[0]
                    in_transit_mask = bls.get_transit_mask(period=period, transit_time=transit_time, duration=duration)
                st.success("Analysis Complete! A high-potential transit signal was detected.")
                st.header("Step 3: Discovery Results & AI Interpretation")
                st.subheader("Key Scientific Evidence")
                c1, c2, c3 = st.columns(3); c1.metric("Orbital Period (days)", f"{period.value:.4f}"); c2.metric("Transit Duration (hours)", f"{(duration.to('h')).value:.2f}"); c3.metric("Transit Depth", f"{depth_value:.4f}")
                st.subheader("🤖 AI Science Communicator's Report")
                with st.spinner("Generating scientific summary..."):
                    explanation_data = {"star_name": lc.label, "period": period.value, "duration": (duration.to('h')).value, "depth": depth_value}
                    explanation = get_llm_explanation(explanation_data)
                    st.write(explanation)
                st.subheader("Visual Proof of Discovery")
                plot_df = lc.to_pandas().reset_index(); plot_df['highlight'] = np.where(in_transit_mask, 'In Transit', 'Out of Transit')
                fig = px.scatter(plot_df, x='time', y='flux', color='highlight', color_discrete_map={'In Transit': 'red', 'Out of Transit': 'blue'}, title=f'Interactive Light Curve for {lc.label}')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"An error occurred: {e}")

elif page == "Manual Data Analysis":
    st.title("🔬 Manual Data Analysis")
    st.write("Upload your own custom data files to classify and analyze a star system.")
    
    if model is None:
        st.error("AI Classifier model (`exoplanet_model_final.h5`) is not loaded. This feature is unavailable.")
    else:
        st.header("Step 1: Upload Your Data Files")
        
        # --- File Uploaders ---
        col1, col2 = st.columns(2)
        with col1:
            features_file = st.file_uploader("Upload Features CSV", type="csv", help="A single-row CSV with the 42 scientific features for the AI Classifier.")
        with col2:
            lightcurve_file = st.file_uploader("Upload Light Curve CSV", type="csv", help="A CSV with two columns ('time', 'flux') for the Discovery Engine.")

        if features_file and lightcurve_file:
            st.success("Both files uploaded! Ready for analysis.")
            
            if st.button("Analyze Uploaded Data", type="primary"):
                # --- Process and Classify with AI Model ---
                st.header("Step 2: AI Classifier Result")
                with st.spinner("Running AI classification..."):
                    features_df = pd.read_csv(features_file)
                    # You would need to ensure the columns match the training data exactly
                    # For this demo, we assume the user provides the correct 42 feature columns
                    features_values = features_df.values
                    features_reshaped = features_values.reshape(1, -1, 1)
                    prediction_prob = model.predict(features_reshaped, verbose=0)[0][0]
                
                res1, res2 = st.columns(2)
                res1.metric("AI Model's Prediction", "Planet Candidate" if prediction_prob > 0.5 else "Not a Planet")
                res2.metric("AI Confidence Score", f"{prediction_prob:.2%}")

                # --- Process and Analyze with Discovery Engine ---
                st.header("Step 3: Discovery Engine Analysis")
                with st.spinner("Analyzing uploaded light curve for transits..."):
                    lc_df = pd.read_csv(lightcurve_file)
                    lc = lk.LightCurve(time=lc_df['time'], flux=lc_df['flux']).remove_nans().normalize().remove_outliers()
                    
                    bls = lc.to_periodogram(method='bls')
                    period = bls.period_at_max_power
                    transit_time = bls.transit_time_at_max_power
                    duration = bls.duration_at_max_power
                    depth_value_raw = bls.depth_at_max_power.value
                    depth_value = depth_value_raw if np.isscalar(depth_value_raw) else depth_value_raw[0]
                    in_transit_mask = bls.get_transit_mask(period=period, transit_time=transit_time, duration=duration)
                
                st.success("Light curve analysis complete!")

                st.subheader("Detected Signal Properties")
                c1, c2, c3 = st.columns(3)
                c1.metric("Orbital Period (days)", f"{period.value:.4f}")
                c2.metric("Transit Duration (hours)", f"{(duration.to('h')).value:.2f}")
                c3.metric("Transit Depth", f"{depth_value:.4f}")

                st.subheader("Visual Proof")
                plot_df = lc.to_pandas().reset_index(); plot_df['highlight'] = np.where(in_transit_mask, 'In Transit', 'Out of Transit')
                fig = px.scatter(plot_df, x='time', y='flux', color='highlight', color_discrete_map={'In Transit': 'red', 'Out of Transit': 'blue'}, title=f'Uploaded Light Curve Analysis')
                st.plotly_chart(fig, use_container_width=True)