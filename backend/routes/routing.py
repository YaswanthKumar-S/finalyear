"""
EV Station Routing API Routes
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
import os
import math

routing_bp = Blueprint('routing', __name__)


def load_stations():
    data_dir = current_app.config['DATA_DIR']
    for fname in ['ev_stations_india.csv', 'ev_stations_chennai.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


@routing_bp.route('/stations', methods=['GET'])
def get_stations():
    """All stations with location + capacity."""
    df = load_stations()
    if df.empty:
        return jsonify([])

    results = []
    for _, r in df.iterrows():
        results.append({
            'station_id': r.get('station_id', ''),
            'name': r.get('station_name', r.get('name', '')),
            'lat': float(r.get('latitude', 0)),
            'lng': float(r.get('longitude', 0)),
            'city': r.get('city', 'Chennai'),
            'area_type': r.get('area_type', 'Metro'),
            'capacity_kw': int(r.get('max_power_kw', r.get('station_capacity_kw', 100))),
            'connectors': int(r.get('num_connectors', r.get('num_chargers', 5))),
            'connector_types': r.get('connector_types', 'CCS2'),
            'operator': r.get('operator_name', r.get('operator', 'Unknown')),
            'is_operational': bool(r.get('is_operational', True)),
            # Simulate availability
            'available_slots': max(0, int(r.get('num_connectors', 5)) - np.random.randint(0, int(r.get('num_connectors', 5)) + 1)),
        })
    return jsonify(results)


@routing_bp.route('/nearest', methods=['GET'])
def get_nearest():
    """Find nearest stations to user location."""
    user_lat = float(request.args.get('lat', 13.0827))
    user_lng = float(request.args.get('lng', 80.2707))
    limit = int(request.args.get('limit', 10))
    max_distance = float(request.args.get('max_km', 50))

    df = load_stations()
    if df.empty:
        return jsonify([])

    results = []
    for _, r in df.iterrows():
        s_lat = float(r.get('latitude', 0))
        s_lng = float(r.get('longitude', 0))
        dist = haversine(user_lat, user_lng, s_lat, s_lng)

        if dist <= max_distance:
            connectors = int(r.get('num_connectors', r.get('num_chargers', 5)))
            available = max(0, connectors - np.random.randint(0, connectors + 1))
            results.append({
                'station_id': r.get('station_id', ''),
                'name': r.get('station_name', r.get('name', '')),
                'lat': s_lat,
                'lng': s_lng,
                'city': r.get('city', 'Chennai'),
                'distance_km': round(dist, 2),
                'capacity_kw': int(r.get('max_power_kw', 100)),
                'connectors': connectors,
                'available_slots': available,
                'connector_types': r.get('connector_types', 'CCS2'),
                'operator': r.get('operator_name', r.get('operator', 'Unknown')),
                'is_operational': bool(r.get('is_operational', True)),
                'eta_min': round(dist / 30 * 60, 0),  # ~30 km/h avg city speed
            })

    results.sort(key=lambda x: x['distance_km'])
    return jsonify(results[:limit])
