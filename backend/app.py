"""
EV Charging Station Management — Flask Backend
================================================
Serves ML predictions and data APIs for the React frontend.
"""

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import os

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Config
    app.config['DATA_DIR'] = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    app.config['MODELS_DIR'] = os.path.join(os.path.dirname(__file__), 'models')

    # JWT Configuration
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'ev-station-jwt-secret-key-2024-secure')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 86400  # 24 hours

    jwt = JWTManager(app)

    # Register blueprints
    from routes.demand import demand_bp
    from routes.location import location_bp
    from routes.routing import routing_bp
    from routes.admin import admin_bp
    from routes.client import client_bp
    from routes.chatbot import chatbot_bp
    from routes.auth import auth_bp

    app.register_blueprint(demand_bp, url_prefix='/api/demand')
    app.register_blueprint(location_bp, url_prefix='/api/location')
    app.register_blueprint(routing_bp, url_prefix='/api/routing')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(client_bp, url_prefix='/api/client')
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')

    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'modules': ['demand', 'location', 'routing', 'admin', 'client', 'chatbot', 'auth']}

    # Seed default admin account
    with app.app_context():
        from routes.auth import seed_admin
        seed_admin()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
