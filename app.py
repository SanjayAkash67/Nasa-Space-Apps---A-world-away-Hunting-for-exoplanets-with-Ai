import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import lightkurve as lk
import warnings
from sklearn.model_selection import train_test_split
import os       # <-- NEW import
import shutil   # <-- NEW import

# --- Page Configuration ---
st.set_page_config(page_title="Exoplanet Explorer", page_icon="🔭", layout="wide")
warnings.filterwarnings('ignore', category=UserWarning)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/exoplanet_model_final.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Main App Interface ---
st.title('🔭 AI Exoplanet Explorer')
st.write("A tool to analyze star data, featuring a trained AI classifier and a live data explorer for NASA's Kepler and TESS missions.")

# --- Create Tabs ---
tab1, tab2 = st.tabs(["🤖 AI Classifier (Demo)", "🛰️ Live Multi-Mission Explorer"])

# ==============================================================================
# TAB 1: AI Classifier (No changes here)
# ==============================================================================
with tab1:
    # ... (The code for Tab 1 remains exactly the same as before) ...
    st.header("Classify a Star from the Kepler Dataset")
    st.write("This tab demonstrates the power of our pre-trained AI model to classify a star system.")
    if model is not None and st.button('Analyze a Random Star', type="primary", key="classify_button"):
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
# TAB 2: Live Multi-Mission Explorer - UPGRADED with Cache Clearing
# ==============================================================================
with tab2:
    st.header("Explore Live Data from NASA's MAST Archive")
    
    mission = st.selectbox("Select Mission:", ("Kepler", "TESS"))

    if mission == "Kepler":
        id_label = "Kepler ID (KIC)"
        default_id = "8462852"
        id_prefix = "KIC"
    else: # TESS
        id_label = "TESS ID (TIC)"
        default_id = "150428135"
        id_prefix = "TIC"

    target_id = st.text_input(f"Enter {id_label}", default_id)

    if st.button("Fetch and Visualize Light Curve", key="fetch_button"):
        if not target_id:
            st.warning(f"Please enter a {id_label}.")
        else:
            try:
                search_string = f"{id_prefix} {target_id}"
                with st.spinner(f"Searching for {search_string} in the {mission} database..."):
                    search_result = lk.search_lightcurve(search_string, mission=mission)
                    if not search_result:
                        st.error(f"No light curve data found for this ID in the {mission} mission.")
                    else:
                        lc_collection = search_result.download_all()
                        lc = lc_collection.stitch().remove_nans().normalize().remove_outliers()
                        st.success(f"Successfully downloaded data for {lc.label}.")
                        fig = px.scatter(x=lc.time.value, y=lc.flux.value, title=f'Interactive Light Curve for {lc.label}', labels={'x': 'Time', 'y': 'Normalized Flux'})
                        fig.update_traces(mode='lines+markers', marker=dict(size=2, opacity=0.7))
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("This might be due to a corrupted file in the cache. Try clearing the cache below.")

    st.write("---")
    st.subheader("Troubleshooting")
    if st.button("Clear Download Cache"):
        try:
            # Find the lightkurve cache directory
            cache_dir = os.path.join(os.path.expanduser('~'), '.lightkurve-cache')
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                st.success(f"Cache cleared successfully from {cache_dir}!")
            else:
                st.warning("Cache directory not found (it may already be clear).")
        except Exception as e:
            st.error(f"Could not clear cache: {e}")