import json
import requests

# ===============================
# ⚡ CONFIGURATION
# ===============================
MLFLOW_SERVER_URL = "http://127.0.0.1:1234/invocations"

# List of features in order expected by the model
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "tenure", "MonthlyCharges", "TotalCharges"
]

def get_input(prompt, expected_type=str):
    """Helper to get user input and convert to the expected type"""
    while True:
        val = input(f"{prompt}: ")
        try:
            return expected_type(val)
        except ValueError:
            print(f"Invalid type, expected {expected_type.__name__}. Try again.")

def main():
    print("=== Interactive Churn Prediction ===")
    print("Please enter the customer details below:")

    # Collect input for all features
    customer = {}
    for f in FEATURES:
        if f in ["SeniorCitizen", "tenure"]:
            customer[f] = get_input(f, int)
        elif f in ["MonthlyCharges", "TotalCharges"]:
            customer[f] = get_input(f, float)
        else:
            customer[f] = get_input(f, str)

    # Prepare request
    payload = {
        "dataframe_split": {
            "columns": FEATURES,
            "data": [list(customer.values())]
        }
    }

    print("🚀 Sending prediction request...")
    response = requests.post(MLFLOW_SERVER_URL, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        pred = response_json["predictions"][0]
        label = "Churn" if pred == 1.0 else "No Churn"
        print(f"📊 Prediction: {label}")
    else:
        print("❌ Request failed with status code:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    main()
