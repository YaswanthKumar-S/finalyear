from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import EVStationPredictor
from datetime import datetime
import logging

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

@app.route('/')
def home():
    return jsonify({
        "message": "🔌 EV Station Location Planning & ROI Prediction API",
        "status": "active" if ev_predictor else "error",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/predict": "POST - Analyze EV station location",
            "/health": "GET - API health check",
            "/models": "GET - List loaded models",
            "/sample": "GET - Sample location data"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if ev_predictor else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(ev_predictor.models) if ev_predictor else 0,
        "service": "EV Station Location Planning"
    })

@app.route('/models', methods=['GET'])
def list_models():
    if not ev_predictor:
        return jsonify({"error": "EV predictor not initialized"}), 500
    
    models_info = ev_predictor.get_model_info()
    return jsonify({
        "models": models_info,
        "total_models": len(models_info),
        "purpose": "EV Station Location Planning and ROI Prediction"
    })

@app.route('/sample', methods=['GET'])
def sample_location():
    """Provide sample location data structure"""
    sample_data = {
        "location_name": "Downtown Business District",
        "daily_vehicles": 15000,
        "population_density": 8500,
        "avg_income": 85000,
        "commercial_score": 85,
        "residential_score": 60,
        "industrial_score": 20,
        "highway_distance": 2.5,
        "mall_distance": 0.5,
        "office_distance": 0.2,
        "ev_adoption": 15,
        "solar_potential": 75,
        "land_cost": 500000,
        "electricity_rate": 0.12,
        "subsidy_available": 30,
        "competition": 2,
        "fast_charging": True,
        "solar_powered": True,
        "amenities": True,
        "high_competition": False,
        "high_land_cost": False
    }
    return jsonify({
        "sample_location": sample_data,
        "description": "Sample data for EV station location analysis",
        "required_fields": ["daily_vehicles", "population_density", "avg_income"]
    })

@app.route('/predict', methods=['POST'])
def predict_ev_station():
    try:
        if not ev_predictor:
            return jsonify({"error": "EV predictor not available"}), 500
        
        location_data = request.get_json()
        
        if not location_data:
            return jsonify({"error": "No location data provided"}), 400
        
        logger.info(f"📍 Analyzing EV station location")
        
        # Validate required fields
        required_fields = ['daily_vehicles', 'population_density', 'avg_income']
        missing_fields = [field for field in required_fields if field not in location_data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        # Make prediction
        result = ev_predictor.predict_ev_station_viability(location_data)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['location_name'] = location_data.get('location_name', 'Unknown Location')
        result['analysis_id'] = f"ev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"✅ EV station analysis completed")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ EV station prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple locations at once"""
    try:
        if not ev_predictor:
            return jsonify({"error": "EV predictor not available"}), 500
        
        data = request.get_json()
        
        if not data or 'locations' not in data:
            return jsonify({"error": "No locations data provided"}), 400
        
        results = []
        for i, location in enumerate(data['locations']):
            try:
                result = ev_predictor.predict_ev_station_viability(location)
                result['location_index'] = i
                result['location_name'] = location.get('location_name', f'Location_{i}')
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "location_index": i,
                    "location_name": location.get('location_name', f'Location_{i}')
                })
        
        return jsonify({
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_locations": len(data['locations']),
            "successful_analysis": len([r for r in results if 'error' not in r]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🔌 Starting EV Station Location Planning API...")
    if ev_predictor:
        print(f"📊 EV models loaded: {len(ev_predictor.models)}")
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)