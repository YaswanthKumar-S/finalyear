/* global google */
import { useState, useEffect, useRef, useCallback } from 'react'
import config from '../config'

const API = '/api/routing'

export default function RouteFinder() {
    const [stations, setStations] = useState([])
    const [nearest, setNearest] = useState([])
    const [userLat, setUserLat] = useState(13.0827)
    const [userLng, setUserLng] = useState(80.2707)
    const [maxKm, setMaxKm] = useState(30)
    const [loading, setLoading] = useState(false)
    const [selectedStation, setSelectedStation] = useState(null)
    const [searchText, setSearchText] = useState('')
    const [showSuggestions, setShowSuggestions] = useState(false)
    const [pickedLocation, setPickedLocation] = useState(null)
    const [routeInfo, setRouteInfo] = useState(null)
    const [mapsLoaded, setMapsLoaded] = useState(false)
    const [routeLoading, setRouteLoading] = useState(false)

    const searchRef = useRef(null)
    const mapRef = useRef(null)
    const mapInstance = useRef(null)
    const directionsRenderer = useRef(null)
    const markersRef = useRef([])

    // Load stations from backend
    useEffect(() => {
        fetch(`${API}/stations`).then(r => r.json()).then(setStations)
    }, [])

    // Load Google Maps script
    useEffect(() => {
        if (window.google && window.google.maps) {
            setMapsLoaded(true)
            return
        }
        const existingScript = document.querySelector('script[src*="maps.googleapis.com"]')
        if (existingScript) {
            existingScript.addEventListener('load', () => setMapsLoaded(true))
            return
        }
        const script = document.createElement('script')
        script.src = `https://maps.googleapis.com/maps/api/js?key=${config.GOOGLE_MAPS_API_KEY}&libraries=places`
        script.async = true
        script.defer = true
        script.onload = () => setMapsLoaded(true)
        document.head.appendChild(script)
    }, [])

    // Initialize Google Map
    useEffect(() => {
        if (!mapsLoaded || !mapRef.current || mapInstance.current) return

        mapInstance.current = new google.maps.Map(mapRef.current, {
            zoom: config.GOOGLE_MAPS_DEFAULT_ZOOM,
            center: config.GOOGLE_MAPS_DEFAULT_CENTER,
            styles: config.GOOGLE_MAPS_DARK_STYLE,
            mapTypeControl: false,
            streetViewControl: false,
        })

        directionsRenderer.current = new google.maps.DirectionsRenderer({
            map: mapInstance.current,
            polylineOptions: {
                strokeColor: '#10b981',
                strokeWeight: 5,
                strokeOpacity: 0.8
            },
            suppressMarkers: false
        })

        // Add user location marker
        new google.maps.Marker({
            position: config.GOOGLE_MAPS_DEFAULT_CENTER,
            map: mapInstance.current,
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 10,
                fillColor: '#8b5cf6',
                fillOpacity: 0.9,
                strokeColor: '#c4b5fd',
                strokeWeight: 3
            },
            title: 'Your Location'
        })
    }, [mapsLoaded])

    // Close suggestions on outside click
    useEffect(() => {
        const handler = (e) => {
            if (searchRef.current && !searchRef.current.contains(e.target))
                setShowSuggestions(false)
        }
        document.addEventListener('mousedown', handler)
        return () => document.removeEventListener('mousedown', handler)
    }, [])

    // Filter stations by search text
    const suggestions = searchText.length > 0
        ? stations.filter(s => {
            const q = searchText.toLowerCase()
            return (s.name?.toLowerCase().includes(q) ||
                s.city?.toLowerCase().includes(q) ||
                s.operator?.toLowerCase().includes(q) ||
                s.area_type?.toLowerCase().includes(q))
        }).slice(0, 8)
        : []

    const pickStation = (s) => {
        setSearchText(s.name)
        setPickedLocation({ name: s.name, city: s.city, lat: s.lat, lng: s.lng })
        setUserLat(s.lat)
        setUserLng(s.lng)
        setShowSuggestions(false)

        if (mapInstance.current) {
            mapInstance.current.panTo({ lat: s.lat, lng: s.lng })
            mapInstance.current.setZoom(13)
        }
    }

    const findNearest = () => {
        setLoading(true)
        fetch(`${API}/nearest?lat=${userLat}&lng=${userLng}&max_km=${maxKm}&limit=15`)
            .then(r => r.json())
            .then(d => {
                setNearest(d)
                setLoading(false)

                // Clear old markers
                markersRef.current.forEach(m => m.setMap(null))
                markersRef.current = []

                if (mapInstance.current && d.length > 0) {
                    // Add station markers
                    d.forEach(s => {
                        const pct = s.available_slots / (s.connectors || 1)
                        const color = pct > 0.5 ? '#10b981' : pct > 0 ? '#f59e0b' : '#ef4444'

                        const marker = new google.maps.Marker({
                            position: { lat: s.lat, lng: s.lng },
                            map: mapInstance.current,
                            icon: {
                                path: google.maps.SymbolPath.CIRCLE,
                                scale: 8,
                                fillColor: color,
                                fillOpacity: 0.8,
                                strokeColor: '#fff',
                                strokeWeight: 2
                            },
                            title: s.name
                        })

                        const infoWindow = new google.maps.InfoWindow({
                            content: `<div style="color:#000;max-width:220px">
                                <strong>${s.name}</strong><br/>
                                📏 ${s.distance_km} km • ⚡ ${s.capacity_kw} kW<br/>
                                Available: ${s.available_slots}/${s.connectors}<br/>
                                ${s.operator}
                            </div>`
                        })

                        marker.addListener('click', () => {
                            infoWindow.open(mapInstance.current, marker)
                            setSelectedStation(s)
                        })

                        markersRef.current.push(marker)
                    })

                    // Add user marker
                    const userMarker = new google.maps.Marker({
                        position: { lat: userLat, lng: userLng },
                        map: mapInstance.current,
                        icon: {
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: 12,
                            fillColor: '#8b5cf6',
                            fillOpacity: 0.9,
                            strokeColor: '#c4b5fd',
                            strokeWeight: 3
                        },
                        title: pickedLocation?.name || 'Your Location'
                    })
                    markersRef.current.push(userMarker)

                    mapInstance.current.panTo({ lat: userLat, lng: userLng })
                    mapInstance.current.setZoom(12)
                }
            })
            .catch(() => setLoading(false))
    }

    const getDirections = useCallback((station) => {
        if (!mapInstance.current || !mapsLoaded) return
        setRouteLoading(true)

        const directionsService = new google.maps.DirectionsService()

        // Reset previous directions
        if (directionsRenderer.current) {
            directionsRenderer.current.setMap(mapInstance.current)
        }

        const trafficLayer = new google.maps.TrafficLayer()
        trafficLayer.setMap(mapInstance.current)

        const request = {
            origin: new google.maps.LatLng(userLat, userLng),
            destination: new google.maps.LatLng(station.lat, station.lng),
            travelMode: google.maps.TravelMode.DRIVING,
            drivingOptions: {
                departureTime: new Date(),
                trafficModel: 'bestguess'
            }
        }

        directionsService.route(request, (response, status) => {
            setRouteLoading(false)
            if (status === google.maps.DirectionsStatus.OK) {
                directionsRenderer.current.setDirections(response)
                const route = response.routes[0]
                const leg = route.legs[0]
                setRouteInfo({
                    distance: leg.distance.text,
                    duration: leg.duration.text,
                    durationInTraffic: leg.duration_in_traffic?.text || leg.duration.text,
                    startAddress: leg.start_address,
                    endAddress: leg.end_address,
                    stationName: station.name
                })
                setSelectedStation(station)
            } else {
                console.error('Directions request failed:', status)
                setRouteInfo({ error: `Could not find route: ${status}` })
            }
        })
    }, [userLat, userLng, mapsLoaded])

    const useMyLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                setUserLat(pos.coords.latitude)
                setUserLng(pos.coords.longitude)
                setSearchText('📱 My Current Location')
                setPickedLocation({ name: 'My Location', city: '', lat: pos.coords.latitude, lng: pos.coords.longitude })
                if (mapInstance.current) {
                    mapInstance.current.panTo({ lat: pos.coords.latitude, lng: pos.coords.longitude })
                    mapInstance.current.setZoom(13)
                }
            })
        }
    }

    const availColor = (available, total) => {
        const pct = available / total
        return pct > 0.5 ? '#10b981' : pct > 0 ? '#f59e0b' : '#ef4444'
    }

    return (
        <div className="page">
            <div className="page-header">
                <h1>🗺️ Route to Nearest Station</h1>
                <p>Find the closest EV charging stations with Google Maps route planning</p>
            </div>

            <div className="grid-2">
                <div>
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">📍 Your Location</span></div>

                        {/* Autosuggest search */}
                        <div ref={searchRef} style={{ position: 'relative', marginBottom: '1rem' }}>
                            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                Search a place or station
                            </label>
                            <input
                                className="form-control"
                                placeholder="Type a station name, city, or area..."
                                value={searchText}
                                onChange={e => { setSearchText(e.target.value); setShowSuggestions(true); setPickedLocation(null) }}
                                onFocus={() => { if (searchText.length > 0) setShowSuggestions(true) }}
                                style={{ fontSize: '0.95rem', padding: '0.75rem 1rem', paddingLeft: '2.25rem' }}
                            />
                            <span style={{ position: 'absolute', left: '0.75rem', top: '2.1rem', fontSize: '1rem', pointerEvents: 'none' }}>📍</span>

                            {pickedLocation && (
                                <div style={{ marginTop: '0.5rem', padding: '0.5rem 0.75rem', background: 'var(--accent-green-dim)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>✓</span>
                                    <span>{pickedLocation.name}{pickedLocation.city ? `, ${pickedLocation.city}` : ''}</span>
                                </div>
                            )}

                            {showSuggestions && suggestions.length > 0 && (
                                <div style={{
                                    position: 'absolute', top: pickedLocation ? 'calc(100% - 1.5rem)' : '100%',
                                    left: 0, right: 0, zIndex: 50, marginTop: '0.25rem',
                                    background: 'var(--bg-secondary)', border: '1px solid var(--border-light)',
                                    borderRadius: 'var(--radius-sm)', boxShadow: 'var(--shadow-lg)',
                                    maxHeight: '300px', overflowY: 'auto'
                                }}>
                                    {suggestions.map((s, i) => (
                                        <div key={i}
                                            onClick={() => pickStation(s)}
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
                                                    ⚡ {s.name}
                                                </div>
                                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.125rem' }}>
                                                    {s.city} • {s.area_type} • {s.operator}
                                                </div>
                                            </div>
                                            <span style={{
                                                fontSize: '0.7rem', fontWeight: 600,
                                                color: availColor(s.available_slots, s.connectors)
                                            }}>
                                                {s.available_slots}/{s.connectors} free
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        <div className="form-group">
                            <label>Max Distance (km)</label>
                            <input type="range" min="5" max="100" value={maxKm}
                                onChange={e => setMaxKm(+e.target.value)} />
                            <div className="slider-label"><span></span><span>{maxKm} km</span></div>
                        </div>
                        <div style={{ display: 'flex', gap: '0.75rem' }}>
                            <button className="btn btn-primary" onClick={findNearest} style={{ flex: 1, justifyContent: 'center' }}>
                                Find Stations
                            </button>
                            <button className="btn btn-secondary" onClick={useMyLocation}>
                                📱 Use GPS
                            </button>
                        </div>
                    </div>

                    {/* Route Info Panel */}
                    {routeInfo && !routeInfo.error && (
                        <div className="card" style={{ marginBottom: '1.5rem', background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1))' }}>
                            <div className="card-header"><span className="card-title">🛣️ Route to {routeInfo.stationName}</span></div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                <div style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Distance</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#10b981' }}>{routeInfo.distance}</div>
                                </div>
                                <div style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Duration</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#3b82f6' }}>{routeInfo.duration}</div>
                                </div>
                                <div style={{ textAlign: 'center', padding: '0.75rem' }}>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>In Traffic</div>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f59e0b' }}>{routeInfo.durationInTraffic}</div>
                                </div>
                            </div>
                        </div>
                    )}
                    {routeInfo?.error && (
                        <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '3px solid var(--accent-red)' }}>
                            <p style={{ color: 'var(--accent-red)', fontSize: '0.85rem' }}>⚠️ {routeInfo.error}</p>
                        </div>
                    )}

                    {/* Station list */}
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {loading && <div className="loading"><div className="spinner" />Searching...</div>}
                        {nearest.map((s, i) => (
                            <div key={i} className="station-card" style={{ marginBottom: '0.75rem', cursor: 'pointer', border: selectedStation?.station_id === s.station_id ? '1px solid var(--accent-green)' : undefined }}
                                onClick={() => {
                                    setSelectedStation(s)
                                    if (mapInstance.current) {
                                        mapInstance.current.panTo({ lat: s.lat, lng: s.lng })
                                        mapInstance.current.setZoom(14)
                                    }
                                }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div>
                                        <div className="name">{s.name}</div>
                                        <div className="meta">
                                            <span>{s.city}</span>
                                            <span>{s.operator}</span>
                                        </div>
                                    </div>
                                    <div className="distance">{s.distance_km} km</div>
                                </div>
                                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', fontSize: '0.8rem', alignItems: 'center' }}>
                                    <div>
                                        <span style={{ color: availColor(s.available_slots, s.connectors), fontWeight: 600 }}>
                                            {s.available_slots}/{s.connectors}
                                        </span> available
                                    </div>
                                    <div>{s.capacity_kw} kW</div>
                                    <div>{s.connector_types}</div>
                                    <div>~{s.eta_min} min</div>
                                    <button className="btn btn-sm btn-primary"
                                        style={{ marginLeft: 'auto', fontSize: '0.7rem', padding: '0.25rem 0.5rem' }}
                                        onClick={(e) => { e.stopPropagation(); getDirections(s) }}
                                        disabled={routeLoading}>
                                        {routeLoading ? '⏳' : '🗺️'} Route
                                    </button>
                                </div>
                            </div>
                        ))}
                        {!loading && nearest.length === 0 && stations.length > 0 && (
                            <div className="card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
                                Search a location and click "Find Stations"
                            </div>
                        )}
                    </div>
                </div>

                {/* Google Map */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">🗺️ Google Maps</span>
                        {routeInfo && !routeInfo.error && (
                            <button className="btn btn-sm btn-secondary" onClick={() => {
                                setRouteInfo(null)
                                if (directionsRenderer.current) {
                                    directionsRenderer.current.setDirections({ routes: [] })
                                }
                            }}>Clear Route</button>
                        )}
                    </div>
                    <div style={{ height: '560px', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                        {!mapsLoaded ? (
                            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                                <div><div className="spinner" style={{ marginBottom: '1rem' }} />Loading Google Maps...</div>
                            </div>
                        ) : (
                            <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
                        )}
                    </div>
                </div>
            </div>

            <div className="stats-grid" style={{ marginTop: '1.5rem' }}>
                <div className="stat-card">
                    <div className="stat-icon green">🏢</div>
                    <div className="stat-info">
                        <h3>{stations.length}</h3>
                        <p>Total Stations</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon blue"></div>
                    <div className="stat-info">
                        <h3>{nearest.length}</h3>
                        <p>Stations Found Nearby</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon orange"></div>
                    <div className="stat-info">
                        <h3>{nearest[0]?.distance_km || '—'} km</h3>
                        <p>Nearest Station</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon green">⚡</div>
                    <div className="stat-info">
                        <h3>{nearest[0]?.available_slots || '—'}</h3>
                        <p>Slots at Nearest</p>
                    </div>
                </div>
            </div>
        </div>
    )
}
