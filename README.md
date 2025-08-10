# 📈 Stock Price Prediction using LSTM & Keras Tuner

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Keras](https://img.shields.io/badge/Keras-TensorFlow-orange?logo=keras)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

A machine learning project that predicts stock prices using **Long Short-Term Memory (LSTM)** networks with **Keras Tuner** for hyperparameter optimization.

🎥 **Demo Video:** [Watch on YouTube](https://youtu.be/e8-Fdg2Jn_o)

---

## 🚀 Features
- 📊 Fetches **historical stock data** from Yahoo Finance.
- 🔧 **Automatic hyperparameter tuning** with Keras Tuner.
- ⏳ Predicts **next-day stock prices** using the past **120 days** of data.
- 🖥 **Streamlit web interface** for interactive predictions & performance metrics.

---

## 📂 Project Structure & Setup
```plaintext
├── train_model.py        # Train and save the LSTM model
├── app.py                # Streamlit UI for predictions
├── requirements.txt      # Required Python packages
├── README.md             # Documentation
└── .gitignore            # Ignore cache & unwanted files
```

---

## 🛠 Installation

### 1️⃣ Clone the Repository
git clone https://github.com/jayavanth18/stock-price-prediction-lstm.git

###2️⃣ Install Dependencies
pip install -r requirements.txt

---

###📊 Usage
python train_model.py
```plaintext
-> Downloads historical data.
-> Tunes the LSTM hyperparameters.
-> Saves the best model to best_stock_model.keras.
```
Run the Streamlit App
streamlit run app.py
```plaintext
-> Enter a stock ticker (e.g., WMT, AAPL).
-> View predicted vs actual prices.
-> Check model accuracy & performance metrics.
```
---

## 📸 Example Output
Prediction Graph's :
<img width="2509" height="1144" alt="Wallmart" src="https://github.com/user-attachments/assets/0b42fac2-2e21-456b-961a-2e3274961b6c" />
<img width="2476" height="1123" alt="Apple" src="https://github.com/user-attachments/assets/0803d304-c860-4d23-8d71-4fe2760908a2" />
<img width="2495" height="1111" alt="Tesla" src="https://github.com/user-attachments/assets/df1d6412-0f0a-4077-8d02-74d421f1141a" />

---

##✨ Author: [A. Jayavanth](https://github.com/jayavanth18)

---
