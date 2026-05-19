import os
import joblib
import numpy as np
import mysql.connector
from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Machine Learning Assets safely using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PredictionData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class AppointmentData(BaseModel):
    fullname: str
    email: str
    phone: str
    appointment_date: str
    doctor: str

@app.post("/api/predict")
def predict_heart_disease(data: PredictionData):
    try:
        input_data = np.array([[ 
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal
        ]])
        
        scaled_data = scaler.transform(input_data)
        prediction = int(model.predict(scaled_data)[0])
        
        # Calculate a simulated probability risk percentage
        probabilities = model.predict_proba(scaled_data)[0]
        risk_percentage = round(float(probabilities[1]) * 100, 2)
        
        label = "High Risk" if prediction == 1 else "Low Risk"
        return {"risk_percentage": risk_percentage, "label": label}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/book-appointment")
def book_appointment(data: AppointmentData):
    try:
        # Automatically parse the single DATABASE_URL into individual connection attributes
        url = urlparse(os.getenv("DATABASE_URL"))
        
        conn = mysql.connector.connect(
            host=url.hostname,
            port=url.port,
            user=url.username,
            password=url.password,
            database=url.path.lstrip('/')
        )
        cursor = conn.cursor()

        # Insert Data securely using parameterized query strings
        sql = """INSERT INTO appointment (patient_name, email, phone, appointment_date, doctor_id)
                 VALUES (%s, %s, %s, %s, %s)"""
        val = (data.fullname, data.email, data.phone, data.appointment_date, data.doctor)

        cursor.execute(sql, val)
        conn.commit()

        return {"status": "success", "message": "Appointment booked successfully in the database!"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Always close the database connection elements to prevent resource leaks
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
