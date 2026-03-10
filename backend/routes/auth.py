"""
Authentication API Routes — Register, Login, and User Info
Uses a JSON file for user storage (no database dependency).
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
import bcrypt
import json
import os
import uuid
from datetime import timedelta

auth_bp = Blueprint('auth', __name__)

USERS_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')


def _load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return []


def _save_users(users):
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def _find_user(email):
    """Find user by email."""
    users = _load_users()
    for u in users:
        if u['email'].lower() == email.lower():
            return u
    return None


def seed_admin():
    """Create default admin account if none exists."""
    users = _load_users()
    admins = [u for u in users if u['role'] == 'admin']
    if not admins:
        hashed = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        users.append({
            'id': str(uuid.uuid4()),
            'name': 'Administrator',
            'email': 'admin@evstation.com',
            'password': hashed,
            'role': 'admin',
        })
        _save_users(users)


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new client account."""
    d = request.json or {}
    name = d.get('name', '').strip()
    email = d.get('email', '').strip()
    password = d.get('password', '')

    if not name or not email or not password:
        return jsonify({'error': 'Name, email, and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400

    if _find_user(email):
        return jsonify({'error': 'An account with this email already exists'}), 409

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    users = _load_users()
    new_user = {
        'id': str(uuid.uuid4()),
        'name': name,
        'email': email.lower(),
        'password': hashed,
        'role': 'client',
    }
    users.append(new_user)
    _save_users(users)

    token = create_access_token(
        identity=new_user['id'],
        additional_claims={'role': 'client', 'name': name, 'email': email.lower()},
        expires_delta=timedelta(hours=24)
    )

    return jsonify({
        'success': True,
        'token': token,
        'user': {'id': new_user['id'], 'name': name, 'email': email.lower(), 'role': 'client'}
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login with email and password."""
    d = request.json or {}
    email = d.get('email', '').strip()
    password = d.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = _find_user(email)
    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = create_access_token(
        identity=user['id'],
        additional_claims={'role': user['role'], 'name': user['name'], 'email': user['email']},
        expires_delta=timedelta(hours=24)
    )

    return jsonify({
        'success': True,
        'token': token,
        'user': {'id': user['id'], 'name': user['name'], 'email': user['email'], 'role': user['role']}
    })


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    """Get current user info from token."""
    claims = get_jwt()
    return jsonify({
        'id': get_jwt_identity(),
        'name': claims.get('name', ''),
        'email': claims.get('email', ''),
        'role': claims.get('role', ''),
    })
