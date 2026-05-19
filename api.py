from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import mysql.connector
import os

# ==========================================
# 1. INITIALIZE THE HOSPITAL SERVER
# ==========================================
app = FastAPI(title="CardioCore Master API")

# Security: Allow your HTML frontend to talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your exported Brain (ML Model)
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# ==========================================
# 2. QUEUE 1: THE MACHINE LEARNING PREDICTOR
# ==========================================
class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/api/predict")
def predict_risk(data: PatientData):
    # Format data for the ML model
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, 
        data.fbs, data.restecg, data.thalach, data.exang, 
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    # Calculate exact probability for the UI Gauge
    probabilities = model.predict_proba(scaled_data)[0]
    risk_percentage = int(probabilities[1] * 100) 
    label = "High Risk" if prediction == 1 else "Low Risk"
    
    return {"risk_percentage": risk_percentage, "label": label}

# ==========================================
# 3. QUEUE 2: THE MYSQL DATABASE CONNECTOR
# ==========================================
class AppointmentData(BaseModel):
    fullname: str
    email: str
    phone: str
    appointment_date: str 
    doctor: str

@app.post("/api/book-appointment")
def book_appointment(data: AppointmentData):
    try:
        # Connect to MySQL Database
      host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
        )
        cursor = conn.cursor()

        # Insert Data securely
        sql = """INSERT INTO appointment (patient_name, email, phone, appointment_date, doctor_id)
                 VALUES (%s, %s, %s, %s, %s)"""
        val = (data.fullname, data.email, data.phone, data.appointment_date, data.doctor)
        
        cursor.execute(sql, val)
        conn.commit()

        return {"status": "success", "message": "Appointment booked successfully in the database!"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Always close the connection to prevent memory leaks
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
