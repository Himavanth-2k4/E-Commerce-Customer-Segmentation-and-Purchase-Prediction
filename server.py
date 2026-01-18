from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import pandas as pd
import os
from typing import Optional
import traceback

app = FastAPI()

# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = []
    for error in errors:
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        error_messages.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error: " + "; ".join(error_messages),
            "prediction": None,
            "status": "error"
        }
    )

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache-busting headers for static files
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Serve static files
if not os.path.exists("client"):
    os.makedirs("client", exist_ok=True)
app.mount("/static", StaticFiles(directory="client"), name="static")

# Paths to model and preprocessing artifacts
model_path = os.path.join("backend", "model", "final_model.pkl")
scaler_path = os.path.join("backend", "model", "scaler.pkl")
city_le_path = os.path.join("backend", "model", "city_le.pkl")
eng_le_path = os.path.join("backend", "model", "eng_le.pkl")
ship_le_path = os.path.join("backend", "model", "ship_le.pkl")
y_le_path = os.path.join("backend", "model", "y_le.pkl")

model = None
scaler = None
city_le = None
eng_le = None
ship_le = None
y_le = None

try:
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully from {model_path}")
    print(f"Model type: {type(model)}")
    if hasattr(model, 'feature_names_in_'):
        print(f"✓ Model feature names: {list(model.feature_names_in_)}")
    else:
        print("Model does NOT have feature_names_in_")
except Exception as e:
    print(f"Error loading model: {e}")

# load optional preprocessing artifacts
try:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
    if os.path.exists(city_le_path):
        city_le = joblib.load(city_le_path)
        print(f"✓ City LabelEncoder loaded from {city_le_path}")
    if os.path.exists(eng_le_path):
        eng_le = joblib.load(eng_le_path)
        print(f"✓ Gender LabelEncoder loaded from {eng_le_path}")
    if os.path.exists(ship_le_path):
        ship_le = joblib.load(ship_le_path)
        print(f"✓ Membership Type LabelEncoder loaded from {ship_le_path}")
    if os.path.exists(y_le_path):
        y_le = joblib.load(y_le_path)
        print(f"✓ Target LabelEncoder (y_le) loaded from {y_le_path}")
except Exception as e:
    print(f"Error loading preprocessing artifacts: {e}")

# ----------------------- INPUT FORMAT -----------------------
# Final columns in EXACT order (matching model's feature_names_in_):
# ['Age', 'Total Spend', 'Items Purchased', 'Average Rating',
#  'Days Since Last Purchase', 'Gender', 'City', 'Membership Type']
class CustomerData(BaseModel):
    Gender: str = Field(..., description="Customer's gender")
    Age: int = Field(..., ge=1, le=120, description="Customer's age in years")
    City: str = Field(..., description="City where the customer is located")
    Membership_Type: str = Field(..., description="Customer's membership tier level")
    Total_Spend: float = Field(..., ge=0, description="Total amount spent by customer")
    Items_Purchased: int = Field(..., ge=0, description="Total number of items purchased")
    Average_Rating: float = Field(..., ge=0, le=5, description="Average rating given by customer (0-5)")
    Days_Since_Last_Purchase: int = Field(..., ge=0, description="Number of days since customer's last purchase")

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

# ----------------------- ROUTES -----------------------
@app.get("/")
async def read_root():
    return FileResponse("client/index.html", media_type="text/html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": {
            "city": city_le is not None,
            "gender": eng_le is not None,
            "membership": ship_le is not None,
            "target": y_le is not None,
        }
    }

@app.get("/index.html")
async def get_index():
    return FileResponse("client/index.html", media_type="text/html")

@app.get("/index.css")
async def get_css():
    return FileResponse("client/index.css", media_type="text/css")

@app.get("/index.js")
async def get_js():
    return FileResponse("client/index.js", media_type="application/javascript")

# ----------------------- PREDICTION ROUTE -----------------------
@app.post("/predict")
async def predict(data: CustomerData):
    if model is None:
        return {"error": "Model not loaded. Please check if the model file exists.", "prediction": None, "status": "error"}

    try:
        # Create DataFrame with the fields as received
        input_data = pd.DataFrame({
            'Gender': [data.Gender],
            'Age': [data.Age],
            'City': [data.City],
            'Membership Type': [data.Membership_Type],
            'Total Spend': [data.Total_Spend],
            'Items Purchased': [data.Items_Purchased],
            'Average Rating': [data.Average_Rating],
            'Days Since Last Purchase': [data.Days_Since_Last_Purchase]
        })

        print(f"Input DataFrame columns: {list(input_data.columns)}")

        # If any preprocessing artifacts are available, apply them safely
        numerical_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
        categorical_cols = ['Gender', 'City', 'Membership Type']

        processed_data = input_data.copy()

        # Safe transform function for LabelEncoders
        def safe_transform(le, val, col_name):
            try:
                if val in le.classes_:
                    return int(le.transform([val])[0])
                else:
                    # fallback to most frequent / first class
                    return int(le.transform([le.classes_[0]])[0])
            except Exception as e:
                print(f"✗ {col_name} encoding error (safe_transform): {e}")
                return 0

        # Encode categorical columns if encoders exist
        if eng_le is not None:
            try:
                processed_data['Gender'] = processed_data['Gender'].apply(lambda x: safe_transform(eng_le, x, 'Gender'))
                print("✓ Applied Gender encoding")
            except Exception as e:
                print(f"✗ Gender encoding error: {e}")

        if city_le is not None:
            try:
                processed_data['City'] = processed_data['City'].apply(lambda x: safe_transform(city_le, x, 'City'))
                print("✓ Applied City encoding")
            except Exception as e:
                print(f"✗ City encoding error: {e}")

        if ship_le is not None:
            try:
                processed_data['Membership Type'] = processed_data['Membership Type'].apply(lambda x: safe_transform(ship_le, x, 'Membership Type'))
                print("✓ Applied Membership Type encoding")
            except Exception as e:
                print(f"✗ Membership Type encoding error: {e}")

        # Scale numerical columns if scaler exists
        if scaler is not None:
            try:
                processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
                print("✓ Applied scaling to numerical features")
            except Exception as e:
                print(f"✗ Scaling error: {e}")

        # Reorder columns to match model.feature_names_in_ if available
        try:
            if hasattr(model, 'feature_names_in_'):
                expected = list(model.feature_names_in_)
                missing = [c for c in expected if c not in processed_data.columns]
                if missing:
                    print(f"Warning: expected feature names missing from processed data: {missing}")
                else:
                    processed_data = processed_data[expected]
                    print("✓ Reordered processed_data to model.feature_names_in_")
        except Exception as e:
            print(f"✗ Reordering to model.feature_names_in_ failed: {e}")

        # Prepare for prediction
        features = processed_data

        # Predict
        try:
            prediction = model.predict(features)
            print("✓ Successfully made prediction")
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            print(f"✗ {error_msg}")
            print(f"DataFrame columns: {list(features.columns)}")
            print(f"DataFrame dtypes: {features.dtypes}")
            raise ValueError(error_msg)

        # Get numeric prediction value
        if isinstance(prediction, (np.ndarray, list, pd.Series)):
            prediction_value = int(prediction[0])
        else:
            prediction_value = int(prediction)

        # Map to label if possible
        satisfaction_label = str(prediction_value)
        if y_le is not None:
            try:
                satisfaction_label = str(y_le.inverse_transform([prediction_value])[0])
            except Exception as e:
                print(f"Note: Could not inverse transform target label: {e}")
        
        # Probabilities
        probabilities = None
        probability_breakdown = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = None
                try:
                    proba = model.predict_proba(features)
                except Exception:
                    try:
                        proba = model.predict_proba(features.values)
                    except Exception:
                        proba = None

                if proba is not None:
                    probabilities = proba[0].tolist()
                    # build label -> percent mapping using y_le if available
                    if y_le is not None and hasattr(y_le, 'classes_'):
                        labels = [str(cls) for cls in y_le.classes_]
                    else:
                        labels = [str(i) for i in range(len(probabilities))]
                    probability_breakdown = {
                        labels[i]: round(probabilities[i] * 100, 2)
                        for i in range(min(len(labels), len(probabilities)))
                    }
        except Exception as e:
            print(f"Note: Could not obtain probabilities: {e}")

        response = {
            "prediction": prediction_value,
            "satisfaction_level": satisfaction_label,
            "status": "success"
        }
        if probabilities is not None:
            response["probabilities"] = probabilities
        if probability_breakdown is not None:
            response["probability_breakdown"] = probability_breakdown

        return response

    except ValueError as ve:
        error_msg = f"Value error: {str(ve)}. Please check your input data format."
        print(f"ValueError: {traceback.format_exc()}")
        return {"error": error_msg, "prediction": None, "status": "error"}
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"Exception: {traceback.format_exc()}")
        return {"error": error_msg, "prediction": None, "status": "error"}