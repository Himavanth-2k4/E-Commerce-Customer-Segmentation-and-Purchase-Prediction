# 🛒 E-Commerce Customer Behavior & Satisfaction Prediction

An end-to-end Machine Learning web application that analyzes e-commerce customer behavior and predicts customer satisfaction levels in real time. The project features a trained scikit-learn model, a high-performance **FastAPI** backend, and an interactive frontend dashboard.

---

## 📌 Project Overview

Understanding customer satisfaction and purchasing patterns is crucial for customer retention and targeted marketing. This project provides:
- **Machine Learning Pipeline:** Trained on customer behavior data (`Age`, `Gender`, `City`, `Membership Type`, `Total Spend`, `Items Purchased`, `Average Rating`, `Days Since Last Purchase`) to classify satisfaction tiers.
- **RESTful API Backend:** Fast, asynchronous prediction API powered by FastAPI with Pydantic request validation and scikit-learn inference.
- **Interactive UI:** Web dashboard to input customer metrics and inspect real-time predictions and probability breakdowns.

---

## 🏗️ Architecture & Flow

```
User Input (Web Dashboard / API)
              │
              ▼
   FastAPI Server (server.py)
   ├── Pydantic Input Validation (CustomerData)
   ├── Label Encoding (Gender, City, Membership Type)
   ├── Standard Scaling (Numerical Features)
   └── Scikit-Learn Model Inference
              │
              ▼
  Satisfaction Prediction & Confidence Breakdown
              │
              ▼
  JSON Response / Dashboard Visualization
```

---

## 📂 Repository Structure

```plaintext
├── backend/
│   ├── model/
│   │   ├── final_model.pkl      # Trained classification model
│   │   ├── scaler.pkl           # Feature scaler
│   │   ├── city_le.pkl          # City label encoder
│   │   ├── eng_le.pkl           # Gender label encoder
│   │   ├── ship_le.pkl          # Membership type label encoder
│   │   └── y_le.pkl             # Target satisfaction label encoder
│   ├── E-commerce Customer Behavior - Sheet1.csv
│   └── Project.ipynb            # Data analysis, EDA, and model training
├── client/
│   ├── index.html               # Frontend dashboard UI
│   ├── index.css                # UI styles and responsive layout
│   └── index.js                 # API caller and DOM updates
├── tests/
│   └── test_server.py           # Unit and integration tests
├── .github/workflows/
│   └── ci.yml                   # CI pipeline for automated testing
├── Dockerfile                   # Container definition
├── requirements.txt             # Project dependencies
├── server.py                    # FastAPI application & prediction endpoints
└── README.md
```

---

## 🛠️ Tech Stack

- **Backend:** Python 3.9+, FastAPI, Uvicorn, Pydantic
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
- **Containerization & CI:** Docker, GitHub Actions

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Himavanth-2k4/E-Commerce-Customer-Segmentation-and-Purchase-Prediction.git
cd E-Commerce-Customer-Segmentation-and-Purchase-Prediction
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to `http://localhost:8000` to interact with the dashboard. Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

---

## 🔌 API Reference

### `POST /predict`
Predicts satisfaction level based on customer transactional metrics.

#### Request Body
```json
{
  "Gender": "Female",
  "Age": 32,
  "City": "San Francisco",
  "Membership_Type": "Gold",
  "Total_Spend": 1250.50,
  "Items_Purchased": 14,
  "Average_Rating": 4.5,
  "Days_Since_Last_Purchase": 12
}
```

#### Response (`200 OK`)
```json
{
  "prediction": 1,
  "satisfaction_level": "Satisfied",
  "status": "success",
  "probabilities": [0.08, 0.85, 0.07],
  "probability_breakdown": {
    "Unsatisfied": 8.0,
    "Satisfied": 85.0,
    "Neutral": 7.0
  }
}
```

---

## 🧪 Testing

Run test suite using `pytest`:
```bash
pytest tests/
```

---

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t ecommerce-customer-prediction .
docker run -p 8000:8000 ecommerce-customer-prediction
```

---

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
