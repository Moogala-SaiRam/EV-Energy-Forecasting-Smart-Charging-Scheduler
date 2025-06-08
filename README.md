A Deep Learning Model for Forecasting Energy Consumption in Electric Vehicles (EVs) to Optimize Charging Schedules
This repository presents a hybrid deep learning-based system that forecasts energy consumption and charging duration for electric vehicles (EVs). It aims to optimize EV charging schedules, alleviate grid stress, and enhance energy management systems using advanced AI techniques.

Project Overview
The project introduces a hybrid model combining:

Gated Recurrent Units (GRU): To capture sequential and temporal dependencies in time-series EV charging data.

Extreme Gradient Boosting (XGBoost): To learn non-linear interactions and complex feature relationships.

These models were trained on an augmented real-world dataset with over 10,000 EV charging sessions containing features like vehicle model, battery state, energy usage, and environmental factors.

Key Features
Hybrid GRU + XGBoost architecture

Time-series prediction of:

Energy Consumption (in kWh)

Charging Duration (in hours)

Performance metrics: MAE, RMSE, MAPE, R² Score

Superior accuracy compared to traditional models (ANN, CNN, LSTM, RNN)

Visual analytics for predicted vs actual results

Project Structure
bash
Copy
Edit
├── ev_charging_patterns_augmented.csv      # Dataset used for training and testing
├── GRU-XGBoost.ipynb                       # Jupyter notebook for GRU-XGBoost implementation
├── Model_Comparision.ipynb                 # Model benchmarking and analysis
├── Sairam_report.docx                      # Full project documentation
└── README.md                               # This file
Results Summary
Metric	GRU-XGBoost (Energy)	GRU-XGBoost (Time)
MAE	5.35 kWh	0.037 hours
RMSE	7.70 kWh	0.056 hours
MAPE	0.32%	0.02%
R² Score	0.8754	0.9970

Requirements
Python 3.7+

TensorFlow / Keras

XGBoost

Scikit-learn

Pandas, NumPy, Matplotlib, Seaborn

Jupyter Notebook or Google Colab

How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/ev-energy-forecasting.git
cd ev-energy-forecasting
Open and run the GRU-XGBoost.ipynb in Jupyter Notebook or Google Colab.

Load the dataset:

ev_charging_patterns_augmented.csv

Run all cells to preprocess data, train the model, and evaluate results.

Reference Dataset
File: ev_charging_patterns_augmented.csv

Size: ~10,000 samples

Fields: User ID, Vehicle Model, Charging Start Time, Charging Duration, Energy Consumed, Battery Info, Environmental Features

Evaluation
The hybrid model demonstrated clear improvements over baselines:

Captures both long-term sequential behavior (GRU) and short-term, high-dimensional interactions (XGBoost)

Robust against outliers and temporal noise

Scalable and suitable for real-world smart grid and EV infrastructure applications

Future Scope
Integrate real-time data from EV sensors and smart chargers

Extend to multi-objective optimization (battery health, dynamic tariffs)

Explore Transformer and attention-based models for improved accuracy

Deploy via FastAPI or Streamlit for real-time monitoring and prediction dashboard
