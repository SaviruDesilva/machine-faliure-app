# 🛠️ Industrial Machine Failure Prediction

A machine learning web application built with **Streamlit** that predicts industrial machine failures. This tool helps businesses reduce downtime and save costs by identifying potential failures before they happen.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machine-faliure-app-apdwknyysmf3flyisezrde.streamlit.app/)

## 📖 Project Overview
This application analyzes sensor data (temperature, rotational speed, torque, tool wear) to classify machines as "Safe" or "At Risk of Failure." It uses a **Support Vector Classifier (SVC)** model trained on historical industrial data.

### Key Features
* **📊 Data Dashboard:** View dataset statistics and distributions (e.g., Temperature vs. Failure).
* **🤖 AI Prediction:** Real-time prediction of machine status (Fail/Safe) using machine learning.
* **💰 Business Cost Analysis:** An interactive calculator that estimates the financial impact of the model by comparing "Maintenance Costs" vs. "Breakdown Costs."
* **📈 Model Performance:** Visualizes the Confusion Matrix and Classification Report to show accuracy.

## 💻 Tech Stack
* **Python 3.x**
* **Streamlit** (Web Interface)
* **Scikit-Learn** (Machine Learning - SVC)
* **Imbalanced-Learn** (SMOTE for handling data imbalance)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)

## 🚀 How to Run Locally

If you want to run this app on your own computer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SaviruDesilva/machine-faliure-app.git](https://github.com/SaviruDesilva/machine-faliure-app.git)
    cd machine-faliure-app
    ```

2.  **Install dependencies:**
    Make sure you have a `requirements.txt` file, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run industry.py
    ```

## 📂 Project Structure
```text
├── industry.csv         # The dataset containing sensor readings
├── industry.py          # The main Streamlit application code
├── requirements.txt     # List of Python libraries required
└── README.md            # Project documentation
