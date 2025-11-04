import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="EV Station Location Planner",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .ev-card {
        background: linear-gradient(135deg, #00b4db, #0083b0);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .roi-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EVStationPlanner:
    def __init__(self):
        self.api_url = "http://localhost:5000"
        self.location_types = ["Urban Center", "Suburban", "Highway", "Commercial", "Residential"]
    
    def check_backend(self):
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def analyze_location(self, location_data):
        """Send location analysis request to backend"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=location_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def setup_sidebar(self):
        """Setup the sidebar with navigation"""
        st.sidebar.title("🔌 EV Station Planner")
        
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Dashboard", "📍 Location Analysis", "📊 Compare Locations", "💰 ROI Calculator"]
        )
        
        st.sidebar.markdown("---")
        
        # Backend status
        if self.check_backend():
            st.sidebar.success("✅ Backend Connected")
        else:
            st.sidebar.error("❌ Backend Offline")
            st.sidebar.info("Start backend: `python ev_app.py`")
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **EV Station Location Planner**
        
        Analyze optimal locations for EV charging stations
        and predict investment returns.
        """)
        
        return page
    
    def show_dashboard(self):
        """Show main dashboard"""
        st.markdown('<h1 class="main-header">🔌 EV Station Location Planner</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Optimal Charging Station Placement & ROI Prediction</p>', unsafe_allow_html=True)
        
        if not self.check_backend():
            st.error("""
            ❌ **Backend API is not available** 
            
            Please start the EV planning backend:
            1. Open terminal in backend folder
            2. Run: `python ev_app.py`
            3. Wait for "Server starting on http://localhost:5000"
            """)
            return
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg ROI", "18-25%", "Premium Locations")
        
        with col2:
            st.metric("Payback Period", "4-7 years", "Typical Range")
        
        with col3:
            st.metric("Station Types", "5", "Location Segments")
        
        with col4:
            st.metric("Success Rate", "92%", "AI Accuracy")
        
        st.markdown("---")
        
        # Quick analysis
        st.subheader("🚀 Quick Location Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **Get instant EV station insights:**
            - Optimal charging station placement
            - ROI prediction and payback period
            - Location-specific recommendations
            - Risk assessment and viability scoring
            - Competitive analysis
            """)
        
        with col2:
            if st.button("🔍 Analyze New Location", use_container_width=True, type="primary"):
                st.session_state.show_quick_analysis = True
                st.rerun()
        
        if st.session_state.get('show_quick_analysis'):
            self.show_quick_analysis()
    
    def show_quick_analysis(self):
        """Show quick location analysis form"""
        st.subheader("📍 Quick Location Assessment")
        
        with st.form("quick_analysis_form"):
            st.write("**Location Basics**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                location_name = st.text_input("Location Name", "Downtown Business District")
                daily_vehicles = st.slider("Daily Vehicle Traffic", 1000, 50000, 15000, 500)
                population_density = st.slider("Population Density (per sq km)", 1000, 20000, 8500, 500)
                avg_income = st.slider("Average Income ($)", 30000, 150000, 85000, 5000)
            
            with col2:
                ev_adoption = st.slider("EV Adoption Rate (%)", 1, 50, 15, 1)
                commercial_score = st.slider("Commercial Activity (0-100)", 0, 100, 85)
                competition = st.slider("Competition Level (1-5)", 1, 5, 2)
                solar_potential = st.slider("Solar Potential (%)", 0, 100, 75)
            
            submitted = st.form_submit_button("🔌 Analyze EV Station Potential", type="primary")
            
            if submitted:
                location_data = {
                    "location_name": location_name,
                    "daily_vehicles": daily_vehicles,
                    "population_density": population_density,
                    "avg_income": avg_income,
                    "commercial_score": commercial_score,
                    "residential_score": 60,
                    "industrial_score": 20,
                    "highway_distance": 2.5,
                    "mall_distance": 0.5,
                    "office_distance": 0.2,
                    "ev_adoption": ev_adoption,
                    "solar_potential": solar_potential,
                    "land_cost": 500000,
                    "electricity_rate": 0.12,
                    "subsidy_available": 30,
                    "competition": competition,
                    "fast_charging": True,
                    "solar_powered": True,
                    "amenities": True,
                    "high_competition": competition >= 4,
                    "high_land_cost": False
                }
                
                with st.spinner("🔍 Analyzing location for optimal EV station placement..."):
                    result = self.analyze_location(location_data)
                
                self.display_ev_analysis(result, location_data)
    
    def show_location_analysis(self):
        """Show detailed location analysis page"""
        st.title("📍 EV Station Location Analysis")
        
        st.info("""
        **Comprehensive EV Station Planning**
        Provide detailed location information for optimal charging station placement
        and accurate ROI predictions.
        """)
        
        with st.form("detailed_analysis_form"):
            st.subheader("📍 Location Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                location_name = st.text_input("Location Name", "Prime Commercial Location")
                location_type = st.selectbox("Location Type", self.location_types)
                daily_vehicles = st.number_input("Daily Vehicle Traffic", 1000, 100000, 15000)
                population_density = st.number_input("Population Density (per sq km)", 1000, 50000, 12000)
                avg_income = st.number_input("Average Annual Income ($)", 30000, 200000, 85000)
            
            with col2:
                ev_adoption = st.number_input("Current EV Adoption Rate (%)", 1, 100, 18)
                land_cost = st.number_input("Land Cost ($)", 100000, 2000000, 500000)
                electricity_rate = st.number_input("Electricity Rate ($/kWh)", 0.05, 0.30, 0.12)
                subsidy_available = st.number_input("Government Subsidy (%)", 0, 50, 25)
            
            st.subheader("🏗️ Location Characteristics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                commercial_score = st.slider("Commercial Activity", 0, 100, 80)
                highway_distance = st.slider("Distance to Highway (km)", 0.1, 20.0, 2.5)
            
            with col2:
                residential_score = st.slider("Residential Activity", 0, 100, 65)
                mall_distance = st.slider("Distance to Mall (km)", 0.1, 10.0, 0.8)
            
            with col3:
                industrial_score = st.slider("Industrial Activity", 0, 100, 25)
                office_distance = st.slider("Distance to Offices (km)", 0.1, 10.0, 0.3)
            
            st.subheader("⚡ EV-Specific Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                solar_potential = st.slider("Solar Potential", 0, 100, 70)
                competition = st.slider("Competition Level (1=Low, 5=High)", 1, 5, 2)
                fast_charging = st.checkbox("Plan for Fast Charging", value=True)
            
            with col2:
                solar_powered = st.checkbox("Include Solar Canopy", value=True)
                amenities = st.checkbox("Include Amenities (Cafe, WiFi)", value=True)
                high_land_cost = st.checkbox("High Land Cost Area")
            
            submitted = st.form_submit_button("🚀 Run Comprehensive Analysis", type="primary")
            
            if submitted:
                location_data = {
                    "location_name": location_name,
                    "daily_vehicles": daily_vehicles,
                    "population_density": population_density,
                    "avg_income": avg_income,
                    "commercial_score": commercial_score,
                    "residential_score": residential_score,
                    "industrial_score": industrial_score,
                    "highway_distance": highway_distance,
                    "mall_distance": mall_distance,
                    "office_distance": office_distance,
                    "ev_adoption": ev_adoption,
                    "solar_potential": solar_potential,
                    "land_cost": land_cost,
                    "electricity_rate": electricity_rate,
                    "subsidy_available": subsidy_available,
                    "competition": competition,
                    "fast_charging": fast_charging,
                    "solar_powered": solar_powered,
                    "amenities": amenities,
                    "high_competition": competition >= 4,
                    "high_land_cost": high_land_cost
                }
                
                with st.spinner("🔍 Running comprehensive EV station analysis..."):
                    result = self.analyze_location(location_data)
                
                self.display_ev_analysis(result, location_data)
    
    def display_ev_analysis(self, result, location_data):
        """Display EV station analysis results"""
        if 'error' in result:
            st.error(f"❌ Analysis failed: {result['error']}")
            return
        
        # Main result card
        st.markdown("---")
        st.markdown('<div class="ev-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            viability = result.get('viability_score', 0)
            st.markdown(f'<h1 style="color: white; font-size: 3rem; text-align: center; margin: 0;">{viability:.0f}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: white; text-align: center; margin: 0;">Viability Score</p>', unsafe_allow_html=True)
            
            roi = result['predictions']['roi']['annual_roi']
            st.markdown(f'<div class="roi-badge">Annual ROI: {roi:.1f}%</div>', unsafe_allow_html=True)
            
            grade = result.get('investment_grade', 'Unknown')
            st.markdown(f'<p style="color: white; text-align: center; font-size: 1.2rem; margin: 0.5rem 0;">{grade}</p>', unsafe_allow_html=True)
            
            risk = result.get('risk_level', 'Unknown')
            st.markdown(f'<p style="color: white; text-align: center; margin: 0;">Risk Level: {risk}</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ROI Details
        st.subheader("💰 Investment Returns")
        
        roi_data = result['predictions']['roi']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Annual ROI", f"{roi_data['annual_roi']:.1f}%")
        
        with col2:
            st.metric("Payback Period", f"{roi_data['payback_period']:.1f} years")
        
        with col3:
            st.metric("Break-even", f"{roi_data['break_even_months']:.0f} months")
        
        with col4:
            st.metric("Est. Annual Revenue", f"${roi_data['estimated_annual_revenue']:,}")
        
        # Location Type
        st.subheader("📍 Location Analysis")
        
        cluster_data = result['predictions']['location_type']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Cluster Type:** {cluster_data['cluster_name']}")
            st.write(f"**Description:** {cluster_data['description']}")
        
        with col2:
            # Create location type gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cluster_data['cluster_id'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Location Tier"},
                gauge = {
                    'axis': {'range': [None, 4]},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 3], 'color': "lightgreen"},
                        {'range': [3, 4], 'color': "green"}
                    ],
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Station Recommendations")
        
        recommendations = result.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Store analysis in session state
        if 'ev_analyses' not in st.session_state:
            st.session_state.ev_analyses = []
        
        st.session_state.ev_analyses.append({
            'timestamp': datetime.now(),
            'location_data': location_data,
            'result': result
        })
        
        st.success("✅ EV station analysis completed successfully!")
    
    def show_compare_locations(self):
        """Show location comparison page"""
        st.title("📊 Compare Locations")
        
        if not self.check_backend():
            st.error("Backend not available. Please start the EV planning backend first.")
            return
        
        if 'ev_analyses' not in st.session_state or len(st.session_state.ev_analyses) == 0:
            st.info("No location analyses yet. Analyze some locations first to compare them.")
            return
        
        analyses = st.session_state.ev_analyses
        
        # Create comparison table
        comparison_data = []
        for analysis in analyses[-5:]:  # Last 5 analyses
            result = analysis['result']
            location_data = analysis['location_data']
            
            comparison_data.append({
                'Location': location_data.get('location_name', 'Unknown'),
                'Viability Score': result.get('viability_score', 0),
                'ROI (%)': result['predictions']['roi']['annual_roi'],
                'Payback (years)': result['predictions']['roi']['payback_period'],
                'Location Type': result['predictions']['location_type']['cluster_name'],
                'Risk Level': result.get('risk_level', 'Unknown')
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # ROI Comparison Chart
        fig = px.bar(
            df, 
            x='Location', 
            y='ROI (%)',
            title='ROI Comparison Across Locations',
            color='ROI (%)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_roi_calculator(self):
        """Show ROI calculator"""
        st.title("💰 ROI Calculator")
        
        st.info("""
        **Quick ROI Estimation**
        Get a quick estimate of EV station returns based on key parameters.
        """)
        
        with st.form("roi_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                installation_cost = st.number_input("Installation Cost ($)", 50000, 500000, 150000, 10000)
                daily_customers = st.slider("Expected Daily Customers", 10, 200, 50, 5)
                charge_price = st.slider("Price per Charge ($)", 10, 50, 25, 5)
            
            with col2:
                operating_cost = st.number_input("Monthly Operating Cost ($)", 1000, 10000, 2500, 500)
                utilization_rate = st.slider("Station Utilization (%)", 10, 100, 60, 5)
                growth_rate = st.slider("Annual Growth Rate (%)", 1, 20, 8, 1)
            
            submitted = st.form_submit_button("📈 Calculate ROI")
            
            if submitted:
                # Simple ROI calculation
                daily_revenue = daily_customers * charge_price * (utilization_rate / 100)
                annual_revenue = daily_revenue * 365
                annual_profit = annual_revenue - (operating_cost * 12)
                
                if installation_cost > 0:
                    roi_percentage = (annual_profit / installation_cost) * 100
                    payback_years = installation_cost / annual_profit
                else:
                    roi_percentage = 0
                    payback_years = float('inf')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Annual ROI", f"{roi_percentage:.1f}%")
                
                with col2:
                    st.metric("Payback Period", f"{payback_years:.1f} years")
                
                with col3:
                    st.metric("Annual Profit", f"${annual_profit:,.0f}")
                
                # Growth projection
                years = 5
                growth_data = []
                current_revenue = annual_revenue
                
                for year in range(1, years + 1):
                    growth_data.append({
                        'Year': year,
                        'Revenue': current_revenue,
                        'Profit': current_revenue - (operating_cost * 12)
                    })
                    current_revenue *= (1 + growth_rate / 100)
                
                growth_df = pd.DataFrame(growth_data)
                fig = px.line(growth_df, x='Year', y='Revenue', title='Revenue Growth Projection')
                st.plotly_chart(fig, use_container_width=True)

def main():
    planner = EVStationPlanner()
    
    # Initialize session state
    if 'show_quick_analysis' not in st.session_state:
        st.session_state.show_quick_analysis = False
    if 'ev_analyses' not in st.session_state:
        st.session_state.ev_analyses = []
    
    # Setup navigation
    page = planner.setup_sidebar()
    
    # Show appropriate page
    if page == "🏠 Dashboard":
        planner.show_dashboard()
    elif page == "📍 Location Analysis":
        planner.show_location_analysis()
    elif page == "📊 Compare Locations":
        planner.show_compare_locations()
    elif page == "💰 ROI Calculator":
        planner.show_roi_calculator()

if __name__ == "__main__":
    main()