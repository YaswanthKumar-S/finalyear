"""
Client Application API Routes — Submit new EV charger installation applications
Enhanced with demand prediction, location suitability, and EV sufficiency analysis.
"""

from flask import Blueprint, request, jsonify, current_app
from routes.auth_middleware import client_required
import pandas as pd
import numpy as np
import os
import uuid
import math
from datetime import datetime

client_bp = Blueprint('client', __name__)


# ── Location scoring (same formula as location.py) ──────────────────────────

def _compute_raw(pop, traf, hwy, comm, exist, opp, pwr, land, ev, inc, park, foot):
    ev_sc = ev / 4.0
    return (pop*1.5 + traf*2.0 + hwy*0.3 + comm*1.5 + opp*1.5 +
            pwr*1.2 + ev_sc*1.0 + inc*1.0 + park*0.5 + foot*1.0 -
            land*1.5 - exist*2.0)

RAW_MIN, RAW_MAX = -35.0, 115.0

def _normalize(raw):
    return (raw - RAW_MIN) / (RAW_MAX - RAW_MIN) * 100


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def _analyze_location(lat, lng, city, area_type):
    """Analyse location suitability and demand using existing data."""
    data_dir = current_app.config['DATA_DIR']
    analysis = {
        'suitability': 'Unknown',
        'optimal_score': 0,
        'demand_estimate': 0,
        'sufficiency_status': 'Unknown',
        'ev_count_area': 0,
        'station_count_area': 0,
        'nearby_stations': 0,
        'recommendation': ''
    }

    # ── Location suitability from planning data ──────────────────────────
    loc_path = None
    for fname in ['ev_location_planning_india.csv', 'ev_location_planning.csv']:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            loc_path = p
            break

    best_match = None
    if loc_path:
        loc_df = pd.read_csv(loc_path)
        # Find nearest location within 10 km
        if 'lat' in loc_df.columns and 'lng' in loc_df.columns:
            loc_df['_dist'] = loc_df.apply(
                lambda r: _haversine(lat, lng, r['lat'], r['lng']), axis=1)
            nearby = loc_df[loc_df['_dist'] < 10].sort_values('_dist')
            if not nearby.empty:
                best_match = nearby.iloc[0]
                analysis['suitability'] = str(best_match.get('suitability', 'Unknown'))
                analysis['optimal_score'] = round(float(best_match.get('optimal_score', 0)), 1)

    # If no match, use formula-based scoring with defaults
    if best_match is None:
        area_scores = {
            'Metro': {'pop': 7, 'traf': 7, 'comm': 7, 'opp': 6, 'pwr': 7, 'exist': 5},
            'Suburban': {'pop': 5, 'traf': 5, 'comm': 5, 'opp': 7, 'pwr': 5, 'exist': 3},
            'Village': {'pop': 3, 'traf': 3, 'comm': 3, 'opp': 8, 'pwr': 4, 'exist': 1},
        }
        s = area_scores.get(area_type, area_scores['Metro'])
        raw = _compute_raw(
            s['pop'], s['traf'], 5, s['comm'], s['exist'], s['opp'],
            s['pwr'], 5, 15, 5, 5, 5)
        score = _normalize(raw)
        analysis['optimal_score'] = round(score, 1)
        if score >= 53.4:
            analysis['suitability'] = 'Highly Suitable'
        elif score >= 47.5:
            analysis['suitability'] = 'Suitable'
        elif score >= 41.5:
            analysis['suitability'] = 'Moderate'
        else:
            analysis['suitability'] = 'Low'

    # ── Demand estimate from demand data ─────────────────────────────────
    for fname in ['ev_demand_india.csv', 'ev_demand_data.csv']:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            demand_df = pd.read_csv(p)
            city_demand = demand_df[demand_df['city'] == city] if 'city' in demand_df.columns else demand_df
            if not city_demand.empty:
                analysis['demand_estimate'] = round(float(city_demand['demand_count'].mean()), 1)
            break

    # ── EV sufficiency analysis ──────────────────────────────────────────
    station_path = None
    for fname in ['ev_stations_india.csv', 'ev_stations_chennai.csv']:
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            station_path = p
            break

    if station_path:
        stn_df = pd.read_csv(station_path)
        # Count stations within 15 km
        if 'latitude' in stn_df.columns and 'longitude' in stn_df.columns:
            stn_df['_dist'] = stn_df.apply(
                lambda r: _haversine(lat, lng, float(r['latitude']), float(r['longitude'])), axis=1)
            nearby_stations = stn_df[stn_df['_dist'] < 15]
            analysis['nearby_stations'] = len(nearby_stations)
            analysis['station_count_area'] = len(nearby_stations)

            # Estimate EV count from location data
            ev_per_1000 = 15  # default
            if best_match is not None and 'ev_per_1000' in best_match.index:
                ev_per_1000 = float(best_match['ev_per_1000'])

            pop_estimate = {'Metro': 150000, 'Suburban': 50000, 'Village': 10000}
            pop = pop_estimate.get(area_type, 50000)
            ev_count = int(pop / 1000 * ev_per_1000)
            analysis['ev_count_area'] = ev_count

            station_count = max(1, len(nearby_stations))
            ratio = ev_count / station_count

            if ratio > 50:
                analysis['sufficiency_status'] = 'Not Sufficient'
                analysis['recommendation'] = (
                    f'High demand area with ~{ev_count} EVs but only {station_count} stations nearby. '
                    'New charging station is highly recommended.')
            elif ratio > 20:
                analysis['sufficiency_status'] = 'Moderately Sufficient'
                analysis['recommendation'] = (
                    f'Moderate coverage with ~{ev_count} EVs and {station_count} stations. '
                    'Additional station would improve service quality.')
            else:
                analysis['sufficiency_status'] = 'Highly Sufficient'
                analysis['recommendation'] = (
                    f'Good coverage with ~{ev_count} EVs and {station_count} stations. '
                    'Area is well-served but additional capacity may still be useful.')
        else:
            analysis['sufficiency_status'] = 'Unknown'
            analysis['recommendation'] = 'Station location data not available for analysis.'

    return analysis


@client_bp.route('/submit', methods=['POST'])
@client_required
def submit_application():
    """Submit a new EV charger installation application."""
    d = request.json
    required = ['applicant', 'email', 'phone', 'property_type', 'locality',
                 'latitude', 'longitude', 'area_sqft', 'parking_spaces',
                 'requested_chargers', 'charger_type']
    missing = [f for f in required if not d.get(f)]
    if missing:
        return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

    app_id = f"EV-APP-{datetime.now().strftime('%Y')}-{uuid.uuid4().hex[:6].upper()}"

    lat = float(d['latitude'])
    lng = float(d['longitude'])
    city = d.get('city', 'Chennai')
    area_type = d.get('area_type', 'Metro')

    # Run location & demand analysis
    analysis = _analyze_location(lat, lng, city, area_type)

    new_app = {
        'application_id': app_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'applicant': d['applicant'],
        'email': d['email'],
        'phone': d['phone'],
        'company': d.get('company', ''),
        'property_type': d['property_type'],
        'city': city,
        'locality': d['locality'],
        'area_type': area_type,
        'latitude': lat,
        'longitude': lng,
        'area_sqft': int(d['area_sqft']),
        'parking_spaces': int(d['parking_spaces']),
        'requested_chargers': int(d['requested_chargers']),
        'charger_type': d['charger_type'],
        'power_kw': int(d.get('power_kw', 50)),
        'has_solar': bool(d.get('has_solar', False)),
        'hours': d.get('hours', '24/7'),
        'status': 'Applied',
        'priority': int(d.get('priority', 5)),
        'updated': datetime.now().strftime('%Y-%m-%d'),
        'suitability': analysis['suitability'],
        'optimal_score': analysis['optimal_score'],
        'sufficiency_status': analysis['sufficiency_status'],
    }

    # Append to CSV
    data_dir = current_app.config['DATA_DIR']
    csv_path = os.path.join(data_dir, 'ev_client_applications.csv')
    df_new = pd.DataFrame([new_app])
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    return jsonify({
        'success': True,
        'application_id': app_id,
        'message': f'Application {app_id} submitted successfully!',
        'status': 'Applied',
        'location_analysis': analysis,
    }), 201


@client_bp.route('/track/<app_id>', methods=['GET'])
def track_application(app_id):
    """Track an application by ID."""
    data_dir = current_app.config['DATA_DIR']
    csv_path = os.path.join(data_dir, 'ev_client_applications.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No applications found'}), 404

    df = pd.read_csv(csv_path)
    match = df[df['application_id'] == app_id]
    if match.empty:
        return jsonify({'error': f'Application {app_id} not found'}), 404

    app = match.iloc[0].fillna('').to_dict()
    return jsonify(app)
