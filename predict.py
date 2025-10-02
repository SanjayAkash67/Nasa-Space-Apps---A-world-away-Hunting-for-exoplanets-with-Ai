import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# Suppress TensorFlow/Keras warnings
warnings.filterwarnings('ignore', category=UserWarning)

def predict_on_sample():
    """
    Loads the trained model and a sample from the dataset,
    then makes a prediction.
    """
    # --- 1. Load Your Trained and Balanced Model ---
    print("Loading the pre-trained model 'exoplanet_model_final.h5'...")
    try:
        model = tf.keras.models.load_model('models/exoplanet_model_final.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Load Data to Make a Prediction ---
    print("Loading and preparing a test sample...")
    df = pd.read_csv('data/data.csv')
    df.fillna(0, inplace=True)
    df_proc = df.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score', 'koi_tce_delivname'])
    df_proc['koi_disposition'] = df_proc['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

    X = df_proc.drop('koi_disposition', axis=1).values
    y = df_proc['koi_disposition'].values

    # Split to get the same test set as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 3. Make a Prediction on a Single Sample ---
    # Let's find a known planet in the test set to demonstrate
    planet_indices = np.where(y_test == 1)[0]
    sample_index = planet_indices[5] # Pick the 6th planet for variety

    single_sample = X_test[sample_index]
    true_label = "Planet" if y_test[sample_index] == 1 else "Not a Planet"

    # IMPORTANT: Reshape the data exactly as you did for training
    single_sample_reshaped = single_sample.reshape(1, -1, 1)

    # Get the model's prediction
    print("Making a prediction...")
    prediction_prob = model.predict(single_sample_reshaped, verbose=0)[0][0]
    prediction_label = "Planet Candidate" if prediction_prob > 0.5 else "Not a Planet"

    print("\n" + "="*30)
    print("      PREDICTION RESULT")
    print("="*30)
    print(f"Testing with a sample known to be a: {true_label}")
    print(f"The model predicts: {prediction_label}")
    print(f"Confidence Score (Probability of being a planet): {prediction_prob:.2%}")
    print("="*30)

if __name__ == "__main__":
    predict_on_sample()