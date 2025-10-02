# 🪐 AI Exoplanet Explorer

![Hackathon](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)

**A complete, end-to-end solution for the NASA Space Apps Challenge that uses AI to accelerate the search for new worlds.**

## 📖 Project Overview

The universe is vast, and NASA's planet-hunting missions like Kepler and TESS have generated an immense amount of data. Sifting through this data to find the tiny, periodic dips in starlight that indicate a transiting exoplanet is a monumental task. This project, the **AI Exoplanet Explorer**, is a powerful tool designed to automate and accelerate this process.

It features a deep learning model trained to classify potential exoplanet candidates and a live, interactive web application that can fetch and visualize data directly from NASA's archives.

## ✨ Key Features

* **🤖 AI Classifier:** A Convolutional Neural Network (CNN) trained on thousands of examples from the Kepler mission. It intelligently analyzes 42 different features of a star system to predict the likelihood of it hosting a planet.
* **🛰️ Live Multi-Mission Explorer:** Connects directly to NASA's Mikulski Archive for Space Telescopes (MAST) to fetch and visualize light curve data for any star in the **Kepler** or **TESS** missions in real-time.
* **📊 Interactive Visualizations:** Uses the Plotly library to generate dynamic, zoomable light curve charts, allowing users to closely inspect potential transit events.
* **🧠 Intelligent Training:** The AI model was trained using advanced techniques like `class_weight` to overcome the natural class imbalance in astronomical data, and `EarlyStopping` to find the optimal performance point and prevent overfitting.
* **⚙️ Robust Error Handling:** Includes a built-in cache-clearing mechanism to handle corrupted data downloads from the NASA archives, ensuring a smooth user experience.

## 🛠️ Tech Stack

* **AI/ML:** TensorFlow, Keras, Scikit-learn
* **Web Framework:** Streamlit
* **Data Access:** Lightkurve, Astroquery
* **Data Handling:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib
* **Development:** Python, VS Code, Google Colab (for GPU training), Git & GitHub

## 🚀 Getting Started

Follow these steps to run the application locally.

**1. Clone the Repository:**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Create and Activate a Virtual Environment:**
```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Dependencies:**
All required libraries are listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Download the Dataset:**
The pre-trained AI classifier uses a dataset from the Kepler mission.
* Download the data from the [NASA Exoplanet Archive on Kaggle](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results).
* Place the `cumulative.csv` file inside the `data/` folder.
* Rename the file to `data.csv`.

**5. Run the Streamlit App:**
```bash
streamlit run app.py
```
The application will open in your web browser!

## 🧠 The AI Model

The heart of this project is a Convolutional Neural Network (CNN).

* **Architecture:** The model uses 1D convolutional layers, which are excellent for finding patterns (like the U-shaped dip of a transit) in time-series data like a light curve.
* **Performance:** The final model achieved a **recall of 93%** on the 'Planet' class. This is a critical metric, as it means the model is highly effective at its primary goal: **finding almost all of the real planets** in the dataset and minimizing the chance that a potential new world is missed.

## 🔮 Future Work

This project has a strong foundation that can be extended even further:

* **Transit Characterization:** Train a new regression model to estimate a planet's physical properties (like its radius or temperature) directly from the light curve.
* **Cloud Deployment:** Deploy the application to a service like Streamlit Community Cloud or Hugging Face Spaces to make it publicly accessible to everyone.
* **Citizen Science Portal:** Expand the tool into a full platform where amateur astronomers can upload and analyze their own telescopic data.

---
*This project was developed for the NASA Space Apps Challenge 2025.*