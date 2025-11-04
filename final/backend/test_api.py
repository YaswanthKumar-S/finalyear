import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    try:
        # Test health endpoint
        print("🧪 Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health")
        print(f"Health: {health_response.json()}")
        
        # Test models endpoint
        print("\n🧪 Testing models endpoint...")
        models_response = requests.get(f"{base_url}/models")
        print(f"Models: {models_response.json()}")
        
        # Test prediction
        print("\n🧪 Testing prediction endpoint...")
        sample_data = {
            "gdp_growth": 7.5,
            "employment_rate": 85.0,
            "infrastructure_score": 75,
            "population_growth": 1.5,
            "region": "South",
            "area_type": "Urban"
        }
        
        prediction_response = requests.post(
            f"{base_url}/predict",
            json=sample_data,
            headers={'Content-Type': 'application/json'}
        )
        
        result = prediction_response.json()
        print(f"Prediction Result:")
        print(f"  Score: {result.get('overall_score', 'N/A')}")
        print(f"  Grade: {result.get('investment_grade', 'N/A')}")
        print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"  Models Used: {result.get('models_used', [])}")
        
        if 'error' in result:
            print(f"  ⚠️  Error: {result['error']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_api()