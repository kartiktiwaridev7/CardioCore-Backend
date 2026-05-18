# CardioCore-Backend
The FastAPI backend and Machine Learning API for CardioCore. Handles real-time clinical heart disease predictions using a trained scikit-learn model and manages secure database connections via MySQL.

# CardioCore Backend API 🧠

This repository contains the backend infrastructure and Machine Learning API for the **CardioCore** project. It serves as the bridge between the clinical frontend interface, the predictive ML models, and the database architecture.

## ⚙️ Tech Stack
* **Framework:** FastAPI (Python)
* **Machine Learning:** scikit-learn, joblib, numpy
* **Database:** MySQL
* **Server:** Uvicorn

## 🚀 Core Features
1. **Machine Learning Queue (`/api/predict`):** Receives 13 clinical indicators from the client, scales the data, and passes it through a trained classification model to return an exact risk percentage and diagnostic label.
2. **Database Queue (`/api/book-appointment`):** securely processes incoming patient data and handles MySQL insertion queries for the hospital's scheduling system.
3. **CORS Configured:** Fully open middleware to accept asynchronous `fetch` requests from the decoupled frontend architecture.

## 📂 File Structure
* `api.py`: The master FastAPI server routing and logic.
* `heart_disease_model.pkl`: The serialized, pre-trained classification model.
* `scaler.pkl`: The data standardization translator.
* `requirements.txt`: Cloud deployment dependencies.
