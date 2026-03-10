"""
Demand Prediction API Routes
"""

from flask import Blueprint, request, jsonify, current_app
from routes.auth_middleware import admin_required
import pandas as pd
import numpy as np
import os

demand_bp = Blueprint('demand', __name__)


def load_demand_data():
    """Load demand CSV data."""
    data_dir = current_app.config['DATA_DIR']
    # Try multi-city first, fall back to chennai-only
    for fname in ['ev_demand_india.csv', 'ev_demand_data.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def load_stations():
    """Load station data."""
    data_dir = current_app.config['DATA_DIR']
    for fname in ['ev_stations_india.csv', 'ev_stations_chennai.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@demand_bp.route('/stations', methods=['GET'])
@admin_required
def get_stations():
    """Get all EV stations."""
    df = load_stations()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient='records'))


@demand_bp.route('/analytics', methods=['GET'])
@admin_required
def get_analytics():
    """Get demand analytics — avg by hour, day, month, and top stations."""
    df = load_demand_data()
    if df.empty:
        return jsonify({'error': 'No demand data found'}), 404

    city = request.args.get('city', None)
    if city:
        df = df[df['city'] == city]

    # Hourly pattern
    hourly = df.groupby('hour')['demand_count'].mean().round(2).to_dict()

    # Daily pattern
    daily = df.groupby('day_of_week')['demand_count'].mean().round(2).to_dict()

    # Monthly trend
    monthly = df.groupby('month')['demand_count'].mean().round(2).to_dict()

    # Top stations
    top_stations = (df.groupby('station_name')['demand_count']
                    .mean().nlargest(10).round(2).to_dict())

    # Area type breakdown
    area_demand = {}
    if 'area_type' in df.columns:
        area_demand = df.groupby('area_type')['demand_count'].mean().round(2).to_dict()

    # City breakdown
    city_demand = {}
    if 'city' in df.columns:
        city_demand = df.groupby('city')['demand_count'].mean().round(2).to_dict()

    # Overall stats
    stats = {
        'total_records': len(df),
        'avg_demand': round(df['demand_count'].mean(), 2),
        'max_demand': int(df['demand_count'].max()),
        'avg_utilization': round(df['utilization_pct'].mean(), 1),
        'avg_energy_kwh': round(df['energy_kwh'].mean(), 2),
    }

    return jsonify({
        'hourly': hourly,
        'daily': daily,
        'monthly': monthly,
        'top_stations': top_stations,
        'area_demand': area_demand,
        'city_demand': city_demand,
        'stats': stats,
    })


@demand_bp.route('/predict', methods=['POST'])
@admin_required
def predict_demand():
    """Predict demand for a station at a given time."""
    data = request.json
    station_id = data.get('station_id')
    hour = int(data.get('hour', 12))
    day_of_week = int(data.get('day_of_week', 0))
    month = int(data.get('month', 1))
    is_weekend = int(day_of_week >= 5)
    is_holiday = int(data.get('is_holiday', 0))
    temperature = float(data.get('temperature', 30))
    is_raining = int(data.get('is_raining', 0))

    # Load station info
    stations = load_stations()
    if stations.empty:
        return jsonify({'error': 'No station data'}), 404

    station = stations[stations['station_id'] == station_id]
    if station.empty:
        return jsonify({'error': f'Station {station_id} not found'}), 404

    station = station.iloc[0]
    num_chargers = int(station.get('num_connectors', station.get('num_chargers', 5)))
    capacity = int(station.get('max_power_kw', station.get('station_capacity_kw', 100)))

    # Try to load XGBoost model
    try:
        import xgboost as xgb
        import joblib
        model_dir = current_app.config['MODELS_DIR']
        model_path = os.path.join(model_dir, 'demand_xgboost_model.json')

        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # Load encoders
            le_station = joblib.load(os.path.join(model_dir, 'le_station.pkl'))
            le_city = joblib.load(os.path.join(model_dir, 'le_city.pkl'))
            le_area = joblib.load(os.path.join(model_dir, 'le_area.pkl'))

            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            # Encode categoricals
            try:
                station_enc = le_station.transform([station_id])[0]
            except ValueError:
                station_enc = 0
            try:
                city_enc = le_city.transform([station.get('city', 'Chennai')])[0]
            except ValueError:
                city_enc = 0
            try:
                area_enc = le_area.transform([station.get('area_type', 'Metro')])[0]
            except ValueError:
                area_enc = 0

            features = np.array([[hour, day_of_week, month, is_weekend, is_holiday,
                                  is_raining, temperature, num_chargers, capacity,
                                  station_enc, city_enc, area_enc,
                                  hour_sin, hour_cos, month_sin, month_cos,
                                  float(station.get('latitude', 13.0)),
                                  float(station.get('longitude', 80.2))]])
            predicted = max(0, round(float(model.predict(features)[0])))
            return jsonify({
                'station_id': station_id,
                'station_name': station.get('station_name', station.get('name', '')),
                'predicted_demand': predicted,
                'max_capacity': num_chargers,
                'utilization_pct': round(predicted / num_chargers * 100, 1) if num_chargers > 0 else 0,
                'model': 'xgboost',
            })
    except Exception as e:
        pass

    # Fallback: heuristic prediction
    base = num_chargers * 0.5
    hour_pattern = [0.1,0.05,0.05,0.05,0.05,0.1,0.2,0.5,0.75,0.85,0.7,0.6,
                    0.55,0.6,0.65,0.7,0.8,0.95,0.9,0.7,0.5,0.3,0.2,0.1]
    predicted = max(0, min(num_chargers, round(num_chargers * hour_pattern[hour])))
    if is_weekend:
        predicted = max(0, round(predicted * 0.8))
    if is_raining:
        predicted = max(0, round(predicted * 0.7))

    return jsonify({
        'station_id': station_id,
        'station_name': station.get('station_name', station.get('name', '')),
        'predicted_demand': predicted,
        'max_capacity': num_chargers,
        'utilization_pct': round(predicted / num_chargers * 100, 1) if num_chargers > 0 else 0,
        'model': 'heuristic',
    })
