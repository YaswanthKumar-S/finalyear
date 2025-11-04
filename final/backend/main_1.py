from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import EVStationPredictor
from datetime import datetime
import logging
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize EV station predictor
try:
    ev_predictor = EVStationPredictor('models')
    logger.info("✅ EV Station Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize EV predictor: {str(e)}")
    ev_predictor = None

# Load LSTM model and scaler
try:
    with open('lstm_scaler.pkl', 'rb') as f:
        lstm_scaler = pickle.load(f)
    lstm_model = load_model('lstm_model.h5')
    logger.info("✅ LSTM model and scaler loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load LSTM model or scaler: {str(e)}")
    lstm_scaler = None
    lstm_model = None

@app.route('/')
def home():
    return jsonify({
        "message": "🔌 EV Station Location Planning & ROI Prediction API",
        "status": "active" if ev_predictor and lstm_model else "error",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/predict": "POST - Analyze EV station location",
            "/predict_demand": "POST - Predict EV demand",
            "/health": "GET - API health check",
            "/models": "GET - List loaded models",
            "/sample": "GET - Sample location data"
        }
    })

@app.route('/predict_demand', methods=['POST'])
def predict_ev_demand():
    try:
        if not lstm_model or not lstm_scaler:
            return jsonify({"error": "LSTM model or scaler not available"}), 500
        
        demand_data = request.get_json()
        
        if not demand_data:
            return jsonify({"error": "No demand data provided"}), 400
        
        logger.info(f"📊 Predicting EV demand")
        
        # Extract and preprocess input data
        required_fields = ['daily_vehicles', 'population_density', 'avg_income', 'ev_adoption']
        missing_fields = [field for field in required_fields if field not in demand_data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        input_data = np.array([[demand_data[field] for field in required_fields]])
        scaled_data = lstm_scaler.transform(input_data)
        scaled_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))  # Reshape for LSTM
        
        # Make prediction
        prediction = lstm_model.predict(scaled_data)
        predicted_demand = lstm_scaler.inverse_transform(prediction)[0][0]  # Inverse scale the prediction
        
        logger.info(f"✅ EV demand prediction completed")
        
        return jsonify({
            "predicted_demand": predicted_demand,
            "timestamp": datetime.now().isoformat(),
            "input_data": demand_data
        })
        
    except Exception as e:
        logger.error(f"❌ EV demand prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("🔌 Starting EV Station Location Planning API...")
    if ev_predictor:
        print(f"📊 EV models loaded: {len(ev_predictor.models)}")
    if lstm_model:
        print("📈 LSTM model for demand prediction loaded")
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)