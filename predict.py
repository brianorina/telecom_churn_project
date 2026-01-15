import requests
import json

# ===============================
# Configuration
# ===============================

MODEL_SERVER_URL = "http://127.0.0.1:1234/invocations"

# Define which columns are categorical and allowed values
CATEGORICAL_ALLOWED = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# ===============================
# Helper functions
# ===============================

def get_input(prompt, expected_type=str, allowed=None):
    while True:
        val = input(f"{prompt}: ")
        try:
            val = expected_type(val)
            if allowed and val not in allowed:
                print(f"Invalid value. Allowed: {allowed}")
                continue
            return val
        except ValueError:
            print(f"Invalid type, expected {expected_type.__name__}. Try again.")

def collect_customer_data():
    customer = {}
    # Categorical fields
    for col, allowed in CATEGORICAL_ALLOWED.items():
        customer[col] = get_input(f"{col}", str, allowed)
    # Numeric fields
    for col in NUMERIC_COLUMNS:
        expected_type = float if col in ["MonthlyCharges", "TotalCharges"] else int
        customer[col] = get_input(f"{col}", expected_type)
    return customer

def predict(customers):
    payload = {
        "dataframe_split": {
            "columns": list(customers[0].keys()),
            "data": [list(c.values()) for c in customers]
        }
    }
    response = requests.post(MODEL_SERVER_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    if response.status_code != 200:
        print("Error:", response.json())
        return
    preds = response.json()["predictions"]
    for i, p in enumerate(preds):
        print(f"Customer {i+1}: {'Churn' if p == 1.0 else 'No Churn'}")

# ===============================
# Main Interactive Loop
# ===============================

if __name__ == "__main__":
    print("=== Interactive Churn Prediction ===")
    customers = []
    while True:
        print("\nEnter details for a new customer:")
        customer = collect_customer_data()
        customers.append(customer)
        more = input("Add another customer? (yes/no): ").strip().lower()
        if more != "yes":
            break

    print("\n🚀 Sending prediction request...")
    predict(customers)
