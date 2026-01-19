# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from pyspark.sql import SparkSession
import mlflow.spark

# ===============================
# 1️⃣ Define API Input Schema
# ===============================
class CustomerData(BaseModel):
    # Add all features your model needs
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ===============================
# 2️⃣ Initialize FastAPI and Spark
# ===============================
app = FastAPI(title="Telecom Churn Prediction API")

spark = SparkSession.builder \
    .appName("ChurnPredictionAPI") \
    .getOrCreate()

# ===============================
# 3️⃣ Load MLflow Registered Model
# ===============================
MODEL_NAME = "Churn_Prediction_Model_Brian"
STAGE = "Staging"  # Or "Production"

model = mlflow.spark.load_model(f"models:/{MODEL_NAME}/{STAGE}")

# ===============================
# 4️⃣ API Endpoints
# ===============================
@app.get("/")
def root():
    return {"message": "Welcome to the Telecom Churn Prediction API!"}

@app.post("/predict")
def predict(customers: List[CustomerData]):
    # Convert input list of Pydantic models to DataFrame
    input_df = pd.DataFrame([customer.dict() for customer in customers])
    
    # Convert pandas DF to Spark DataFrame
    spark_df = spark.createDataFrame(input_df)
    
    # Run prediction
    predictions = model.transform(spark_df)
    
    # Collect results
    result_df = predictions.select("prediction").toPandas()
    result_list = result_df["prediction"].astype(int).tolist()
    
    return {"predictions": result_list}
