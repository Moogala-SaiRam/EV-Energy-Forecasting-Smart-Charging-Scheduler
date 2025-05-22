# AI-Based EV Energy Forecasting & Smart Charging Scheduler using Deep Learning

This project is a full-stack intelligent EV forecasting platform that uses deep learning (CNN + GRU) to predict future energy consumption, optimize charging times, and guide users via a smart dashboard. It includes data ingestion, AI modeling, API development, and frontend integration — all deployed with production-ready best practices.

## Features
- Predict energy usage for EVs based on user input (vehicle, station, time, temperature, etc.)
- Smart charging schedule suggestions to minimize cost and avoid grid overload
- Analytics for users and managers: usage trends, busiest hours, model accuracy
- Admin panel for uploading new data and retraining the model
- Secure API endpoints (JWT authentication for admin)

## Tech Stack
- **Model**: Python, TensorFlow/Keras, Pandas, NumPy
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Deployment**: Docker (optional)

## Project Structure
- `src/` — AI models, API backend
- `dashboard/` — Streamlit frontend
- `data/` — Datasets
- `notebooks/` — Data exploration and experiments
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## Quickstart
1. Install requirements: `pip install -r requirements.txt`
2. Train the model: `python src/model_train.py`
3. Start the API: `uvicorn src.api_app:app --reload --port 8000`
4. Run the dashboard: `streamlit run dashboard/dashboard_app.py`

---

*See the full story and user journey in the project documentation.*
