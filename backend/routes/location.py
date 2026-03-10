"""
Location Planner API Routes
"""

from flask import Blueprint, request, jsonify, current_app
from routes.auth_middleware import admin_required
import pandas as pd
import numpy as np
import os

location_bp = Blueprint('location', __name__)


def load_location_data():
    data_dir = current_app.config['DATA_DIR']
    for fname in ['ev_location_planning_india.csv', 'ev_location_planning.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


# Scoring formula (same as training notebook)
W = {'pop': 1.5, 'traf': 2.0, 'hwy': 0.3, 'comm': 1.5, 'opp': 1.5,
     'pwr': 1.2, 'ev': 1.0, 'inc': 1.0, 'park': 0.5, 'foot': 1.0,
     'land': -1.5, 'exist': -2.0}

RAW_MIN = -35.0  # all positive=0, all negative=10
RAW_MAX = 115.0  # all positive=10, all negative=0


def compute_raw(pop, traf, hwy, comm, exist, opp, pwr, land, ev, inc, park, foot):
    ev_sc = ev / 4.0
    return (pop*1.5 + traf*2.0 + hwy*0.3 + comm*1.5 + opp*1.5 +
            pwr*1.2 + ev_sc*1.0 + inc*1.0 + park*0.5 + foot*1.0 -
            land*1.5 - exist*2.0)


def normalize(raw):
    return (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100


@location_bp.route('/all', methods=['GET'])
@admin_required
def get_all():
    """All scored locations."""
    df = load_location_data()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient='records'))


@location_bp.route('/predict', methods=['POST'])
@admin_required
def predict_suitability():
    """Predict suitability from score inputs."""
    d = request.json

    pop = float(d.get('pop_score', 5))
    traf = float(d.get('traffic_score', 5))
    hwy = float(d.get('highway_score', 5))
    comm = float(d.get('commercial_score', 5))
    exist = float(d.get('existing_density', 5))
    opp = float(d.get('opportunity_score', 5))
    pwr = float(d.get('power_grid_score', 5))
    land = float(d.get('land_cost', 5))
    ev = float(d.get('ev_per_1000', 15))
    inc = float(d.get('income_index', 5))
    park = float(d.get('parking_score', 5))
    foot = float(d.get('footfall_score', 5))
    lat = float(d.get('lat', 13.0))
    lng = float(d.get('lng', 80.2))

    # Try ML model first
    try:
        import joblib
        model_dir = current_app.config['MODELS_DIR']
        rf = joblib.load(os.path.join(model_dir, 'location_rf_model.pkl'))
        le = joblib.load(os.path.join(model_dir, 'location_label_encoder.pkl'))

        features = np.array([[pop, traf, hwy, comm, exist, opp, pwr, land, ev, inc, park, foot, lat, lng]])
        pred = rf.predict(features)
        proba = rf.predict_proba(features)
        label = le.inverse_transform(pred)[0]
        confidence = float(proba.max())

        raw = compute_raw(pop, traf, hwy, comm, exist, opp, pwr, land, ev, inc, park, foot)
        score = normalize(raw)

        return jsonify({
            'suitability': label,
            'confidence': round(confidence * 100, 1),
            'score': round(score, 1),
            'model': 'random_forest',
        })
    except Exception:
        pass

    # Fallback: formula-based
    raw = compute_raw(pop, traf, hwy, comm, exist, opp, pwr, land, ev, inc, park, foot)
    score = normalize(raw)

    # Percentile thresholds from training
    if score >= 53.4:
        label = 'Highly Suitable'
    elif score >= 47.5:
        label = 'Suitable'
    elif score >= 41.5:
        label = 'Moderate'
    else:
        label = 'Low'

    return jsonify({
        'suitability': label,
        'confidence': 85.0,
        'score': round(score, 1),
        'model': 'formula',
    })


@location_bp.route('/clusters', methods=['GET'])
@admin_required
def get_clusters():
    """Get clustered location data."""
    df = load_location_data()
    if df.empty:
        return jsonify([])

    # Group by suitability
    result = {}
    for suit in ['Highly Suitable', 'Suitable', 'Moderate', 'Low']:
        subset = df[df['suitability'] == suit]
        result[suit] = {
            'count': len(subset),
            'avg_score': round(subset['optimal_score'].mean(), 2) if len(subset) > 0 else 0,
            'locations': subset[['location', 'lat', 'lng', 'optimal_score']].head(20).to_dict('records') if 'location' in subset.columns else []
        }
    return jsonify(result)
