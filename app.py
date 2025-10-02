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

# --- API Key Configuration ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, AttributeError):
    st.warning("⚠️ Gemini API key not found. The AI Science Communicator will be disabled.")
    genai = None

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/exoplanet_model_final.h5')
        return model
    except Exception:
        return None

# --- LLM Science Communicator Function ---
@st.cache_data
def get_llm_explanation(_data_dict):
    if not genai:
        return "AI Science Communicator is disabled (API key not configured)."
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are an expert astronomer for the NASA Space Apps Challenge.
    Explain the results of an exoplanet transit search for a star named '{_data_dict['star_name']}' in an exciting, one-paragraph summary.

    Data discovered:
    - Orbital Period: {_data_dict['period']:.4f} days
    - Transit Duration: {_data_dict['duration']:.2f} hours
    - Transit Depth: {_data_dict['depth']:.4f}

    Explain these numbers simply (e.g., orbital period is the planet's "year", depth relates to its size).
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "Could not generate explanation at this time."

# --- Main App ---
model = load_model()
st.title('🚀 AI Exoplanet Discovery Engine')
st.write("An advanced tool for exoplanet discovery, combining a trained AI classifier with a live, automated transit detection system for NASA's Kepler and TESS missions.")
tab1, tab2 = st.tabs(["🤖 AI Classifier (Demo)", "🛰️ Automated Discovery Engine (Live)"])

# ==============================================================================
# TAB 1: AI Classifier
# ==============================================================================
with tab1:
    st.header("Classify a Star from the Kepler Dataset")
    if model is None:
        st.error("AI model `exoplanet_model_final.h5` not found.")
    elif st.button('Analyze a Random Star', type="primary", key="classify_button"):
        df = pd.read_csv('data/data.csv')
        df.fillna(0, inplace=True)
        df_display = df.copy()
        df_proc = df.drop(columns=['rowid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score', 'koi_tce_delivname'])
        df_proc['koi_disposition'] = df_proc['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)
        X = df_proc.drop(['koi_disposition', 'kepid'], axis=1).values
        y = df_proc['koi_disposition'].values
        kepids = df_proc['kepid'].values
        _, X_test, _, y_test, _, kepids_test = train_test_split(
            X, y, kepids, test_size=0.2, random_state=np.random.randint(0, 1000), stratify=y)
        sample_index = np.random.randint(0, len(X_test))
        kepid_of_sample = kepids_test[sample_index]
        true_label = "Planet" if y_test[sample_index] == 1 else "Not a Planet"
        st.subheader(f"Analyzing Star KIC {kepid_of_sample} (Ground Truth: {true_label})")
        star_data_full = df_display[df_display['kepid'] == kepid_of_sample].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Orbital Period (days)", f"{star_data_full['koi_period']:.2f}")
        col2.metric("Transit Duration (hours)", f"{star_data_full['koi_duration']:.2f}")
        col3.metric("Transit Depth (ppm)", f"{star_data_full['koi_depth']:.2f}")
        col4.metric("Planet's Temp (K)", f"{star_data_full['koi_teq']:.0f}")
        single_sample_reshaped = X_test[sample_index].reshape(1, -1, 1)
        prediction_prob = model.predict(single_sample_reshaped, verbose=0)[0][0]
        res1, res2 = st.columns(2)
        res1.metric("AI Model's Prediction", "Planet Candidate" if prediction_prob > 0.5 else "Not a Planet")
        res2.metric("AI Confidence Score", f"{prediction_prob:.2%}")

# ==============================================================================
# TAB 2: Automated Discovery Engine - FINAL
# ==============================================================================
with tab2:
    st.header("Automated Transit Search in NASA's Live Archives")
    st.info("This analysis can take 1-2 minutes as it involves downloading and processing real scientific data.")
    
    mission = st.selectbox("Select Mission:", ("Kepler", "TESS"))
    
    if mission == "Kepler":
        id_label, default_id, id_prefix = "Kepler ID (KIC)", "6541920", "KIC"
    else:
        id_label, default_id, id_prefix = "TESS ID (TIC)", "150428135", "TIC"

    target_id = st.text_input(f"Enter {id_label}", default_id)

    if st.button("Search for Transits!", key="fetch_button", type="primary"):
        if not target_id:
            st.warning(f"Please enter a {id_label}.")
        else:
            try:
                search_string = f"{id_prefix} {target_id}"
                
                with st.spinner(f"1/3: Searching for {search_string}..."):
                    search_result = lk.search_lightcurve(search_string, mission=mission)
                    if not search_result:
                        st.error(f"No data found for this ID in {mission}.")
                        st.stop()
                
                with st.spinner(f"2/3: Downloading a subset of data..."):
                    lc_collection = search_result[0:5].download_all()
                    lc = lc_collection.stitch().remove_nans().normalize().remove_outliers()

                with st.spinner("3/3: Analyzing light curve for transit signals..."):
                    bls = lc.to_periodogram(method='bls')
                    period = bls.period_at_max_power
                    transit_time = bls.transit_time_at_max_power
                    duration = bls.duration_at_max_power
                    
                    # --- THIS IS THE DEFINITIVE FIX ---
                    # Get the raw numerical value of the depth
                    depth_value_raw = bls.depth_at_max_power.value
                    # Check if this raw value is a scalar (a single number)
                    if np.isscalar(depth_value_raw):
                        depth_value = depth_value_raw
                    else:
                        # If it's an array, take the first element
                        depth_value = depth_value_raw[0]
                    # ------------------------------------

                    in_transit_mask = bls.get_transit_mask(period=period, transit_time=transit_time, duration=duration)

                st.success("Analysis complete! A potential transit signal was detected.")

                st.subheader("Detected Signal Properties")
                c1, c2, c3 = st.columns(3)
                c1.metric("Orbital Period (days)", f"{period.value:.4f}")
                c2.metric("Transit Duration (hours)", f"{(duration.to('h')).value:.2f}")
                c3.metric("Transit Depth", f"{depth_value:.4f}")

                st.subheader("🤖 AI Science Communicator's Report")
                explanation_data = {"star_name": lc.label, "period": period.value, "duration": (duration.to('h')).value, "depth": depth_value}
                explanation = get_llm_explanation(explanation_data)
                st.write(explanation)

                st.subheader("Interactive Light Curve with Detected Transits Highlighted")
                plot_df = lc.to_pandas().reset_index(); plot_df['highlight'] = np.where(in_transit_mask, 'In Transit', 'Out of Transit')
                fig = px.scatter(plot_df, x='time', y='flux', color='highlight', color_discrete_map={'In Transit': 'red', 'Out of Transit': 'blue'}, title=f'Interactive Light Curve for {lc.label}')
                fig.update_traces(marker=dict(size=3, opacity=0.8))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("This might be due to a network issue or a problem with the data file. Try another ID or try clearing the cache below.")

    st.write("---")
    st.subheader("Troubleshooting")
    if st.button("Clear Download Cache"):
        try:
            cache_dir = lk.get_cache_dir()
            shutil.rmtree(cache_dir)
            st.success("Cache cleared successfully!")
        except Exception as e:
            st.error(f"Could not clear cache: {e}")