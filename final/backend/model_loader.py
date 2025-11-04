import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EVStationPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.load_ev_models()
    
    def load_ev_models(self):
        """Load EV station prediction models"""
        try:
            print("🔌 Loading EV Station Planning Models...")
            
            # Load essential models with fallbacks
            self.load_essential_models()
            
            # Create feature mappings for EV stations
            self.ev_features = [
                'traffic_density', 'population_density', 'income_level', 
                'commercial_score', 'residential_score', 'industrial_score',
                'proximity_highway', 'proximity_mall', 'proximity_office',
                'ev_adoption_rate', 'solar_potential', 'land_cost',
                'electricity_cost', 'government_subsidy', 'competition_score'
            ]
            
            print(f"✅ EV models loaded: {len(self.models)}")
            
        except Exception as e:
            print(f"❌ Error loading EV models: {e}")
    
    def load_essential_models(self):
        """Load essential models with fallbacks"""
        # Load XGBoost model for ROI prediction
        try:
            xgb_path = os.path.join(self.model_dir, 'xgb_model.json')
            if os.path.exists(xgb_path):
                self.models['roi_predictor'] = xgb.Booster()
                self.models['roi_predictor'].load_model(xgb_path)
                print("✅ ROI predictor loaded")
        except Exception as e:
            print(f"❌ ROI predictor failed: {e}")
            self.create_fallback_roi_model()
        
        # Load cluster model for location segmentation
        try:
            cluster_path = os.path.join(self.model_dir, 'cluster_model.pkl')
            if os.path.exists(cluster_path):
                with open(cluster_path, 'rb') as f:
                    self.models['location_cluster'] = pickle.load(f)
                print("✅ Location cluster model loaded")
        except Exception as e:
            print(f"❌ Cluster model failed: {e}")
            self.create_fallback_cluster_model()
    
    def create_fallback_roi_model(self):
        """Create fallback ROI prediction model"""
        # Simple ROI calculation based on key factors
        self.models['roi_predictor'] = 'fallback'
        print("✅ Created fallback ROI model")
    
    def create_fallback_cluster_model(self):
        """Create fallback location clustering"""
        from sklearn.cluster import KMeans
        self.models['location_cluster'] = KMeans(n_clusters=4, random_state=42)
        print("✅ Created fallback cluster model")
    
    def predict_ev_station_viability(self, location_data):
        """Predict EV station viability and ROI"""
        try:
            print(f"📍 Analyzing location: {location_data.get('location_name', 'Unknown')}")
            
            # Prepare features
            features = self.prepare_ev_features(location_data)
            
            predictions = {}
            
            # ROI Prediction
            roi_prediction = self.predict_roi(features, location_data)
            predictions['roi'] = roi_prediction
            
            # Location Cluster
            cluster_prediction = self.predict_location_cluster(features)
            predictions['location_type'] = cluster_prediction
            
            # Station Recommendations
            recommendations = self.generate_station_recommendations(
                roi_prediction, cluster_prediction, location_data
            )
            
            result = {
                'predictions': predictions,
                'recommendations': recommendations,
                'viability_score': self.calculate_viability_score(roi_prediction, cluster_prediction),
                'investment_grade': self.get_investment_grade(roi_prediction['annual_roi']),
                'risk_level': self.get_risk_level(roi_prediction['annual_roi'])
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'error': str(e)}
    
    def prepare_ev_features(self, location_data):
        """Prepare EV-specific features"""
        features = {}
        
        # Traffic and population features
        features['traffic_density'] = location_data.get('daily_vehicles', 0) / 1000
        features['population_density'] = location_data.get('population_density', 0)
        features['income_level'] = location_data.get('avg_income', 0) / 1000
        
        # Location type scores
        features['commercial_score'] = location_data.get('commercial_score', 0)
        features['residential_score'] = location_data.get('residential_score', 0)
        features['industrial_score'] = location_data.get('industrial_score', 0)
        
        # Proximity features
        features['proximity_highway'] = location_data.get('highway_distance', 10)
        features['proximity_mall'] = location_data.get('mall_distance', 5)
        features['proximity_office'] = location_data.get('office_distance', 3)
        
        # EV-specific features
        features['ev_adoption_rate'] = location_data.get('ev_adoption', 0)
        features['solar_potential'] = location_data.get('solar_potential', 0)
        features['land_cost'] = location_data.get('land_cost', 0) / 100000
        features['electricity_cost'] = location_data.get('electricity_rate', 0)
        features['government_subsidy'] = location_data.get('subsidy_available', 0)
        features['competition_score'] = location_data.get('competition', 0)
        
        return features
    
    def predict_roi(self, features, location_data):
        """Predict ROI for EV station"""
        try:
            if self.models['roi_predictor'] != 'fallback':
                # Use XGBoost model
                feature_array = np.array([list(features.values())])
                dmatrix = xgb.DMatrix(feature_array)
                base_roi = float(self.models['roi_predictor'].predict(dmatrix)[0])
            else:
                # Fallback ROI calculation
                base_roi = self.calculate_fallback_roi(features)
            
            # Adjust ROI based on additional factors
            adjusted_roi = self.adjust_roi(base_roi, location_data)
            
            return {
                'annual_roi': adjusted_roi,
                'payback_period': self.calculate_payback_period(adjusted_roi),
                'break_even_months': self.calculate_break_even(adjusted_roi),
                'estimated_annual_revenue': self.estimate_revenue(location_data)
            }
            
        except Exception as e:
            print(f"❌ ROI prediction failed: {e}")
            return {
                'annual_roi': 15.0,  # Default ROI
                'payback_period': 6.7,
                'break_even_months': 80,
                'estimated_annual_revenue': 120000
            }
    
    def calculate_fallback_roi(self, features):
        """Fallback ROI calculation"""
        base_roi = 12.0  # Base 12% ROI
        
        # Adjust based on key factors
        if features['traffic_density'] > 5:
            base_roi += 3
        if features['income_level'] > 75:
            base_roi += 2
        if features['ev_adoption_rate'] > 20:
            base_roi += 4
        if features['competition_score'] < 3:
            base_roi += 2
        if features['government_subsidy'] > 0:
            base_roi += 3
            
        return min(base_roi, 35)  # Cap at 35%
    
    def adjust_roi(self, base_roi, location_data):
        """Adjust ROI based on location-specific factors"""
        adjusted = base_roi
        
        # Positive adjustments
        if location_data.get('fast_charging', False):
            adjusted += 5
        if location_data.get('solar_powered', False):
            adjusted += 3
        if location_data.get('amenities', False):
            adjusted += 2
        
        # Negative adjustments
        if location_data.get('high_competition', False):
            adjusted -= 4
        if location_data.get('high_land_cost', False):
            adjusted -= 3
            
        return max(adjusted, 5)  # Minimum 5% ROI
    
    def predict_location_cluster(self, features):
        """Predict location cluster type"""
        try:
            feature_array = np.array([list(features.values())])
            cluster = self.models['location_cluster'].predict(feature_array)[0]
            
            cluster_types = {
                0: "Premium Urban - High traffic, high income",
                1: "Commercial Hub - Shopping centers, offices",
                2: "Residential Area - Steady local traffic", 
                3: "Highway Corridor - Travel and transit focus",
                4: "Developing Area - Growth potential"
            }
            
            return {
                'cluster_id': int(cluster),
                'cluster_name': cluster_types.get(cluster, "Unknown"),
                'description': self.get_cluster_description(cluster)
            }
            
        except Exception as e:
            print(f"❌ Cluster prediction failed: {e}")
            return {
                'cluster_id': 2,
                'cluster_name': "Standard Commercial",
                'description': "Balanced location with moderate potential"
            }
    
    def get_cluster_description(self, cluster_id):
        """Get description for cluster type"""
        descriptions = {
            0: "Excellent location with high EV adoption potential and premium pricing capability",
            1: "Strong commercial traffic with good revenue potential during business hours",
            2: "Consistent residential usage with potential for loyalty programs",
            3: "High-volume transient traffic, ideal for fast charging stations",
            4: "Emerging area with growth potential, lower initial returns but high future upside"
        }
        return descriptions.get(cluster_id, "Standard location with balanced potential")
    
    def generate_station_recommendations(self, roi_prediction, cluster_prediction, location_data):
        """Generate station-specific recommendations"""
        recommendations = []
        
        # ROI-based recommendations
        roi = roi_prediction['annual_roi']
        if roi > 25:
            recommendations.append("🚀 **Premium Investment** - Consider multiple charging bays")
        elif roi > 15:
            recommendations.append("✅ **Strong Investment** - Optimal for single station")
        elif roi > 8:
            recommendations.append("🔄 **Moderate Investment** - Start with basic infrastructure")
        else:
            recommendations.append("⏸️ **Evaluate Carefully** - Consider alternative locations")
        
        # Cluster-based recommendations
        cluster_id = cluster_prediction['cluster_id']
        if cluster_id == 0:  # Premium Urban
            recommendations.append("💎 **Install DC Fast Chargers** - Target premium customers")
            recommendations.append("🏢 **Add Lounge Amenities** - Increase dwell time revenue")
        elif cluster_id == 1:  # Commercial Hub
            recommendations.append("🛒 **Partner with Retailers** - Cross-promotion opportunities")
            recommendations.append("⏰ **Focus on Business Hours** - Peak pricing strategy")
        elif cluster_id == 3:  # Highway Corridor
            recommendations.append("⚡ **Ultra-Fast Charging** - Minimize stop time for travelers")
            recommendations.append("🍔 **Add Food Services** - Capture additional revenue")
        
        # Feature-based recommendations
        if location_data.get('solar_potential', 0) > 70:
            recommendations.append("☀️ **Add Solar Canopy** - Reduce electricity costs and carbon footprint")
        
        if location_data.get('ev_adoption', 0) < 10:
            recommendations.append("📊 **Community Education** - Work with local EV groups to drive adoption")
        
        return recommendations
    
    def calculate_viability_score(self, roi_prediction, cluster_prediction):
        """Calculate overall viability score (0-100)"""
        roi_score = min(roi_prediction['annual_roi'] * 2, 50)  # ROI contributes up to 50 points
        
        cluster_scores = {0: 45, 1: 40, 2: 30, 3: 35, 4: 25}
        cluster_score = cluster_scores.get(cluster_prediction['cluster_id'], 20)
        
        return min(roi_score + cluster_score, 100)
    
    def calculate_payback_period(self, annual_roi):
        """Calculate payback period in years"""
        if annual_roi <= 0:
            return float('inf')
        return 100 / annual_roi
    
    def calculate_break_even(self, annual_roi):
        """Calculate break-even in months"""
        payback_years = self.calculate_payback_period(annual_roi)
        return payback_years * 12
    
    def estimate_revenue(self, location_data):
        """Estimate annual revenue"""
        base_revenue = 50000  # Base annual revenue
        vehicles = location_data.get('daily_vehicles', 1000)
        ev_rate = location_data.get('ev_adoption', 5)
        
        estimated_ev_traffic = vehicles * (ev_rate / 100) * 0.1  # 10% conversion
        revenue = base_revenue + (estimated_ev_traffic * 365 * 15)  # $15 per charge
        
        return int(revenue)
    
    def get_investment_grade(self, roi):
        """Get investment grade based on ROI"""
        if roi >= 25: return "A+ (Excellent)"
        elif roi >= 20: return "A (Very Good)"
        elif roi >= 15: return "B+ (Good)"
        elif roi >= 10: return "B (Average)"
        elif roi >= 5: return "C (Fair)"
        else: return "D (Poor)"
    
    def get_risk_level(self, roi):
        """Get risk level based on ROI"""
        if roi >= 20: return "Low Risk"
        elif roi >= 12: return "Moderate Risk"
        elif roi >= 8: return "High Risk"
        else: return "Very High Risk"
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {}
        for name, model in self.models.items():
            info[name] = {
                'type': type(model).__name__ if hasattr(model, '__class__') else str(type(model)),
                'status': 'loaded'
            }
        return info