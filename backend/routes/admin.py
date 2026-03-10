"""
Admin Panel API Routes
Enhanced with deployed stations map endpoint and CSV persistence for status updates.
"""

from flask import Blueprint, request, jsonify, current_app
from routes.auth_middleware import admin_required
import pandas as pd
import numpy as np
import os
import math

admin_bp = Blueprint('admin', __name__)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def load_applications():
    data_dir = current_app.config['DATA_DIR']
    path = os.path.join(data_dir, 'ev_client_applications.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_stations():
    data_dir = current_app.config['DATA_DIR']
    for fname in ['ev_stations_india.csv', 'ev_stations_chennai.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def _compute_sufficiency(row):
    """Compute EV sufficiency status for an application row."""
    if 'sufficiency_status' in row and row.get('sufficiency_status') and str(row.get('sufficiency_status')) != 'nan':
        return str(row['sufficiency_status'])

    # Fallback: compute from location data
    try:
        lat = float(row.get('latitude', 0))
        lng = float(row.get('longitude', 0))
        area_type = row.get('area_type', 'Metro')
    except (ValueError, TypeError):
        return 'Unknown'

    if lat == 0 and lng == 0:
        return 'Unknown'

    stn_df = load_stations()
    if stn_df.empty or 'latitude' not in stn_df.columns:
        return 'Unknown'

    stn_df['_dist'] = stn_df.apply(
        lambda r: _haversine(lat, lng, float(r['latitude']), float(r['longitude'])), axis=1)
    nearby = len(stn_df[stn_df['_dist'] < 15])

    pop_estimate = {'Metro': 150000, 'Suburban': 50000, 'Village': 10000}
    pop = pop_estimate.get(area_type, 50000)
    ev_count = int(pop / 1000 * 15)
    station_count = max(1, nearby)
    ratio = ev_count / station_count

    if ratio > 50:
        return 'Not Sufficient'
    elif ratio > 20:
        return 'Moderately Sufficient'
    return 'Highly Sufficient'


@admin_bp.route('/applications', methods=['GET'])
@admin_required
def get_applications():
    """List all applications with filtering."""
    df = load_applications()
    if df.empty:
        return jsonify({'applications': [], 'total': 0, 'page': 1, 'per_page': 20, 'pages': 1})

    # Add sufficiency if missing
    if 'sufficiency_status' not in df.columns:
        df['sufficiency_status'] = ''

    # Filters
    status = request.args.get('status', None)
    city = request.args.get('city', None)
    search = request.args.get('search', None)

    if status and status != 'all':
        df = df[df['status'] == status]
    if city and city != 'all':
        df = df[df['city'] == city] if 'city' in df.columns else df
    if search:
        search_lower = search.lower()
        mask = (df['applicant'].str.lower().str.contains(search_lower, na=False) |
                df['locality'].str.lower().str.contains(search_lower, na=False) |
                df['application_id'].str.lower().str.contains(search_lower, na=False))
        df = df[mask]

    # Sort
    sort_by = request.args.get('sort', 'date')
    ascending = request.args.get('order', 'desc') == 'asc'
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    # Pagination
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    total = len(df)
    df = df.iloc[(page-1)*per_page : page*per_page]

    return jsonify({
        'applications': df.fillna('').to_dict(orient='records'),
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
    })


@admin_bp.route('/applications/<app_id>', methods=['PUT'])
@admin_required
def update_application(app_id):
    """Update application status — persists to CSV."""
    data = request.json
    new_status = data.get('status')
    if not new_status:
        return jsonify({'error': 'status required'}), 400

    valid = ['Applied', 'Under Review', 'Site Visit Scheduled', 'Site Visited',
             'Approved', 'Installation In Progress', 'Deployed', 'Rejected']
    if new_status not in valid:
        return jsonify({'error': f'Invalid status. Valid: {valid}'}), 400

    # Persist to CSV
    data_dir = current_app.config['DATA_DIR']
    csv_path = os.path.join(data_dir, 'ev_client_applications.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        mask = df['application_id'] == app_id
        if mask.any():
            df.loc[mask, 'status'] = new_status
            df.loc[mask, 'updated'] = pd.Timestamp.now().strftime('%Y-%m-%d')
            df.to_csv(csv_path, index=False)

    return jsonify({'application_id': app_id, 'status': new_status, 'updated': True})


@admin_bp.route('/deployed', methods=['GET'])
@admin_required
def get_deployed_stations():
    """Get all deployed/approved applications for map display."""
    df = load_applications()
    if df.empty:
        return jsonify([])

    deployed = df[df['status'].isin(['Deployed', 'Approved', 'Installation In Progress'])]
    if deployed.empty:
        return jsonify([])

    results = []
    for _, r in deployed.iterrows():
        try:
            lat = float(r.get('latitude', 0))
            lng = float(r.get('longitude', 0))
        except (ValueError, TypeError):
            continue
        if lat == 0 and lng == 0:
            continue
        results.append({
            'application_id': r.get('application_id', ''),
            'applicant': r.get('applicant', ''),
            'locality': r.get('locality', ''),
            'city': r.get('city', ''),
            'lat': lat,
            'lng': lng,
            'status': r.get('status', ''),
            'property_type': r.get('property_type', ''),
            'requested_chargers': int(r.get('requested_chargers', 0)),
            'charger_type': r.get('charger_type', ''),
        })

    return jsonify(results)


@admin_bp.route('/stats', methods=['GET'])
@admin_required
def get_stats():
    """Dashboard statistics."""
    apps = load_applications()
    stations = load_stations()

    stats = {
        'total_applications': len(apps),
        'total_stations': len(stations),
        'status_breakdown': {},
        'city_breakdown': {},
        'property_types': {},
        'monthly_apps': {},
    }

    if not apps.empty:
        stats['status_breakdown'] = apps['status'].value_counts().to_dict()
        if 'city' in apps.columns:
            stats['city_breakdown'] = apps['city'].value_counts().to_dict()
        stats['property_types'] = apps['property_type'].value_counts().to_dict()
        if 'date' in apps.columns:
            apps['month'] = pd.to_datetime(apps['date'], errors='coerce').dt.month
            stats['monthly_apps'] = apps['month'].dropna().astype(int).value_counts().sort_index().to_dict()
        stats['pending'] = len(apps[apps['status'].isin(['Applied', 'Under Review'])])
        stats['approved'] = len(apps[apps['status'].isin(['Approved', 'Deployed', 'Installation In Progress'])])
        stats['rejected'] = len(apps[apps['status'] == 'Rejected'])
        stats['deployed'] = len(apps[apps['status'] == 'Deployed'])

    return jsonify(stats)
