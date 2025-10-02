# 🪐 AI Exoplanet Explorer

**A submission for the NASA Space Apps Challenge 2025.**

This project uses a deep learning model to analyze star data from NASA's Kepler and TESS missions, helping to identify potential exoplanet candidates. The project features a trained AI classifier and a live data explorer that can fetch and visualize data directly from NASA's archives.

## ✨ Features

* **🤖 AI Classifier:** A Convolutional Neural Network (CNN) trained on the Kepler dataset to classify stars as "Planet Candidate" or "Not a Planet".
* **🛰️ Live Multi-Mission Explorer:** An interactive tool to fetch and visualize light curve data for any star from the Kepler (KIC) or TESS (TIC) missions in real-time.
* **⚙️ Error Handling:** Includes a built-in cache-clearing mechanism to handle corrupted data downloads.
* **📊 Interactive Visualizations:** Uses Plotly to create zoomable, pannable charts of light curve data.

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/SanjayAkash67/Nasa-Space-Apps---A-world-away-Hunting-for-exoplanets-with-Ai.git
cd nasa-space-app