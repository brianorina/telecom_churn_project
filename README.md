This project is an end-to-end customer churn prediction system built using Apache Spark (PySpark) and Spark ML Pipelines.
The goal is to predict whether a telecom customer is likely to churn (Yes/No) based on demographic, service usage, and billing information.
The project is intentionally structured to reflect real-world ML engineering practices, including:
Modular code organization
Reproducible preprocessing
Spark ML Pipelines
Clear separation between data logic and execution
Foundations for MLOps (to be added)


🧠 Why Spark?
Handles large-scale datasets
Distributed processing
Built-in ML Pipelines for safe, repeatable transformations
Industry-standard for data engineering & ML platforms
Even though the dataset here is small, the architecture is production-ready.

telecom_churn_project/
├── train.py                  # Entry point: trains & evaluates the model
├── test_preprocessing.py     # Test runner for data preprocessing
├── src/
│   ├── __init__.py           # Marks src as a Python package
│   ├── preprocessing.py     # Data loading & cleaning logic
│   └── config.py             # Centralized configuration/constants
├── spark_env/                # Python virtual environment
└── README.md


⚙️ Environment
Python: 3.9.6
PySpark: 4.0.1
OS: macOS
Model: Logistic Regression (Spark ML)



📥 Dataset
Telco Customer Churn Dataset
Source: IBM / Kaggle (loaded directly via URL)
Key preprocessing steps:
Schema inference
Handling invalid TotalCharges
Dropping corrupted rows
Creating binary target label (label)


🧪 Data Preprocessing
src/preprocessing.py
main function :: load_and_clean_data(spark)



Responsibilities:
Downloads dataset using SparkFiles
Cleans numeric columns
Creates binary label:
1 → Churn
0 → No churn
Returns a clean Spark DataFrame
This module is not meant to be run directly — it is imported by training/testing scripts.

🚀 How to Run the Project
1️⃣ Activate virtual environment
source spark_env/bin/activate


2️⃣ Test preprocessing only
python test_preprocessing.py
This verifies:
Spark starts correctly
Data loads successfully
Cleaning logic works

3️⃣ Train the model
python train.py

This will:
Load and preprocess data
Encode categorical features
Assemble feature vectors
Train Logistic Regression
Evaluate model performance


🧱 Machine Learning Pipeline (Spark ML)
The training pipeline includes:
StringIndexer
Converts categorical strings → numeric indices
OneHotEncoder
Avoids false ordinality in categorical features
VectorAssembler
Combines all features into a single vector column
Logistic Regression
Binary classifier predicting churn probability
All steps are combined using a Spark ML Pipeline, ensuring:
Consistent transformations
No data leakage
Reproducibility

