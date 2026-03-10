import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../AuthContext'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { Bar } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Tooltip } from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip)

const API = '/api/location'
const SLIDERS = [
    { key: 'pop_score', label: 'Population Density', icon: '👥' },
    { key: 'traffic_score', label: 'Traffic Volume', icon: '🚗' },
    { key: 'highway_score', label: 'Highway Proximity', icon: '🛣️' },
    { key: 'commercial_score', label: 'Commercial Activity', icon: '🏢' },
    { key: 'existing_density', label: 'Existing Station Density', icon: '📍', inverted: true },
    { key: 'opportunity_score', label: 'Growth Opportunity', icon: '📈' },
    { key: 'power_grid_score', label: 'Power Grid Access', icon: '⚡' },
    { key: 'land_cost', label: 'Land Cost (higher=expensive)', icon: '💰', inverted: true },
    { key: 'income_index', label: 'Income Level', icon: '💎' },
    { key: 'parking_score', label: 'Parking Availability', icon: '🅿️' },
    { key: 'footfall_score', label: 'Foot Traffic', icon: '🚶' },
]

const SUIT_COLORS = {
    'Highly Suitable': '#10b981', 'Suitable': '#3b82f6',
    'Moderate': '#f59e0b', 'Low': '#ef4444'
}
const SUIT_BADGE_CLS = {
    'Highly Suitable': 'badge-green', 'Suitable': 'badge-blue',
    'Moderate': 'badge-orange', 'Low': 'badge-red'
}

export default function LocationPlanner() {
    const { getAuthHeaders } = useAuth()
    const [locations, setLocations] = useState([])
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(true)
    const [searchText, setSearchText] = useState('')
    const [showSuggestions, setShowSuggestions] = useState(false)
    const [selectedLoc, setSelectedLoc] = useState(null)
    const searchRef = useRef(null)
    const [scores, setScores] = useState({
        pop_score: 5, traffic_score: 5, highway_score: 5, commercial_score: 5,
        existing_density: 3, opportunity_score: 7, power_grid_score: 5,
        land_cost: 5, ev_per_1000: 15, income_index: 5,
        parking_score: 5, footfall_score: 5, lat: 13.0, lng: 80.2
    })

    useEffect(() => {
        fetch(`${API}/all`, { headers: getAuthHeaders() }).then(r => r.json()).then(d => {
            setLocations(d)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    // Close suggestions on outside click
    useEffect(() => {
        const handler = (e) => {
            if (searchRef.current && !searchRef.current.contains(e.target))
                setShowSuggestions(false)
        }
        document.addEventListener('mousedown', handler)
        return () => document.removeEventListener('mousedown', handler)
    }, [])

    // Filter locations by search text
    const suggestions = searchText.length > 0
        ? locations.filter(l => {
            const q = searchText.toLowerCase()
            return (l.location?.toLowerCase().includes(q) ||
                l.city?.toLowerCase().includes(q) ||
                l.area_type?.toLowerCase().includes(q))
        }).slice(0, 8)
        : locations.slice(0, 8)

    const pickLocation = (loc) => {
        setSearchText(`${loc.location}, ${loc.city}`)
        setSelectedLoc(loc)
        setShowSuggestions(false)
        setScores(prev => ({ ...prev, lat: loc.lat, lng: loc.lng }))
    }

    const predict = () => {
        const locationName = searchText || 'Custom Location'
        fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify({ ...scores, location_name: locationName })
        }).then(r => r.json()).then(d => setPrediction({ ...d, location_name: locationName }))
    }

    if (loading) return <div className="loading"><div className="spinner" />Loading location data...</div>

    const suitBadge = (label) => {
        const cls = SUIT_BADGE_CLS[label] || 'badge-red'
        return <span className={`badge ${cls}`}>{label}</span>
    }

    return (
        <div className="page">
            <div className="page-header">
                <h1>📍 Optimal Location Planner</h1>
                <p>Evaluate and predict the best locations for new EV charging stations</p>
            </div>

            <div className="grid-2">
                <div className="card" style={{ maxHeight: '700px', overflow: 'auto' }}>
                    <div className="card-header"><span className="card-title">Score Parameters</span></div>

                    {/* Autocomplete search */}
                    <div ref={searchRef} style={{ marginBottom: '1rem', position: 'relative' }}>
                        <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            Search Location
                        </label>
                        <input
                            className="form-control"
                            placeholder="Type a location name, city, or area type..."
                            value={searchText}
                            onChange={e => { setSearchText(e.target.value); setShowSuggestions(true); setSelectedLoc(null) }}
                            onFocus={() => setShowSuggestions(true)}
                            style={{ fontSize: '0.95rem', padding: '0.75rem 1rem', paddingLeft: '2.25rem' }}
                        />
                        <span style={{ position: 'absolute', left: '0.75rem', top: '2.1rem', fontSize: '1rem', pointerEvents: 'none' }}>📍</span>
                        {selectedLoc && (
                            <div style={{ marginTop: '0.5rem', padding: '0.5rem 0.75rem', background: 'var(--accent-green-dim)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>✓ Selected:</span>
                                <span>{selectedLoc.location}, {selectedLoc.city}</span>
                                {suitBadge(selectedLoc.suitability)}
                            </div>
                        )}

                        {/* Suggestions dropdown */}
                        {showSuggestions && suggestions.length > 0 && (
                            <div style={{
                                position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
                                marginTop: selectedLoc ? '-2rem' : '0.25rem',
                                background: 'var(--bg-secondary)', border: '1px solid var(--border-light)',
                                borderRadius: 'var(--radius-sm)', boxShadow: 'var(--shadow-lg)',
                                maxHeight: '320px', overflowY: 'auto'
                            }}>
                                {suggestions.map((loc, i) => (
                                    <div key={i}
                                        onClick={() => pickLocation(loc)}
                                        style={{
                                            padding: '0.625rem 0.875rem', cursor: 'pointer',
                                            borderBottom: '1px solid var(--border)',
                                            transition: 'background 0.15s',
                                            display: 'flex', alignItems: 'center', justifyContent: 'space-between'
                                        }}
                                        onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-card-hover)'}
                                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                    >
                                        <div>
                                            <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>
                                                📍 {loc.location}
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.125rem' }}>
                                                {loc.city} • {loc.area_type} • Score: {loc.optimal_score?.toFixed(1)}
                                            </div>
                                        </div>
                                        {suitBadge(loc.suitability)}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {SLIDERS.map(s => (
                        <div key={s.key} style={{ marginBottom: '0.75rem' }}>
                            <div className="slider-label">
                                <span style={{ fontSize: '0.8rem' }}>{s.icon} {s.label}</span>
                                <span>{scores[s.key]}</span>
                            </div>
                            <input type="range" min="0" max="10" step="0.5" value={scores[s.key]}
                                onChange={e => setScores({ ...scores, [s.key]: +e.target.value })} />
                        </div>
                    ))}
                    <div className="form-group" style={{ marginTop: '0.5rem' }}>
                        <label>EV per 1000 people</label>
                        <input type="range" min="0" max="40" step="1" value={scores.ev_per_1000}
                            onChange={e => setScores({ ...scores, ev_per_1000: +e.target.value })} />
                        <div className="slider-label"><span></span><span>{scores.ev_per_1000}</span></div>
                    </div>
                    <button className="btn btn-primary" onClick={predict} style={{ width: '100%', justifyContent: 'center', marginTop: '0.5rem' }}>
                        🔮 Predict Suitability
                    </button>
                </div>

                <div>
                    {prediction && (
                        <div className="prediction-result" style={{ marginBottom: '1.5rem' }}>
                            <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                📍 {prediction.location_name}
                            </p>
                            <div className="label">{suitBadge(prediction.suitability)}</div>
                            <div className="score">{prediction.score}</div>
                            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                                out of 100 • Confidence: {prediction.confidence}%
                            </p>
                            <p className="confidence">Model: <span className="badge badge-green">{prediction.model}</span></p>
                        </div>
                    )}
                    <div className="card">
                        <div className="card-header"><span className="card-title">Location Map</span></div>
                        <div className="map-container">
                            <MapContainer center={[15, 78]} zoom={5} style={{ height: '100%', width: '100%' }}>
                                <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                                    attribution='&copy; CARTO' />
                                {locations.map((loc, i) => (
                                    <CircleMarker key={i}
                                        center={[loc.lat, loc.lng]}
                                        radius={Math.max(4, (loc.optimal_score || 50) / 12)}
                                        pathOptions={{
                                            color: SUIT_COLORS[loc.suitability] || '#666',
                                            fillOpacity: 0.6
                                        }}>
                                        <Popup>
                                            <strong>{loc.location}</strong><br />
                                            {loc.city && <span>{loc.city} • </span>}
                                            Score: {loc.optimal_score?.toFixed(1)}<br />
                                            {loc.suitability}
                                        </Popup>
                                    </CircleMarker>
                                ))}
                            </MapContainer>
                        </div>
                    </div>
                </div>
            </div>

            <div className="card" style={{ marginTop: '1.5rem' }}>
                <div className="card-header"><span className="card-title">All Scored Locations</span></div>
                <div className="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Location</th><th>City</th><th>Area Type</th>
                                <th>Score</th><th>Suitability</th><th>ROI %</th>
                            </tr>
                        </thead>
                        <tbody>
                            {locations.slice(0, 30).map((loc, i) => (
                                <tr key={i}>
                                    <td style={{ fontWeight: 500 }}>{loc.location}</td>
                                    <td>{loc.city || '—'}</td>
                                    <td><span className="badge badge-purple">{loc.area_type || '—'}</span></td>
                                    <td style={{ fontWeight: 600 }}>{loc.optimal_score?.toFixed(1)}</td>
                                    <td>{suitBadge(loc.suitability)}</td>
                                    <td>{loc.annual_roi_pct?.toFixed(1)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
