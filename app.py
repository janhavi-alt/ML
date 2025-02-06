#python.exe -m pip install fastapi joblib scikit-learn uvicorn
#python -m uvicorn app:app --reload

from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Initialize FastAPI app
app = FastAPI()

# Train and Save the Model (Only Runs Once)
def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.pkl")
    print("✅ Model trained and saved as model.pkl")

# Train and save the model (only if model.pkl doesn't exist)
import os
if not os.path.exists("model.pkl"):
    train_and_save_model()

# Load the trained model
model = joblib.load("model.pkl")

# Define input data format
class PredictionRequest(BaseModel):
    features: list  # Expecting a list of numerical features

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the ML Model API"}

# Prediction endpoint
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Convert input to NumPy array and reshape
        input_data = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Run the app using: uvicorn app:app --reload  Use FastAPI's Interactive Docs
'''Open your browser.
Go to http://127.0.0.1:8000/docs.
Click on the POST /predict/ endpoint.
Click Try it out.
Enter the following JSON in the request body:
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
Click Execute and see the response below.'''


