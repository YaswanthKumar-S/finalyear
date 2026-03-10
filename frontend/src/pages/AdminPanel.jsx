import { useState, useEffect } from 'react'
import { useAuth } from '../AuthContext'
import { Bar, Doughnut } from 'react-chartjs-2'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import {
    Chart as ChartJS, CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Tooltip, Legend)

const API = '/api/admin'
const STATUSES = ['Applied', 'Under Review', 'Site Visit Scheduled', 'Site Visited',
    'Approved', 'Installation In Progress', 'Deployed', 'Rejected']

const STATUS_BADGE = {
    'Applied': 'badge-blue', 'Under Review': 'badge-orange',
    'Site Visit Scheduled': 'badge-purple', 'Site Visited': 'badge-purple',
    'Approved': 'badge-green', 'Installation In Progress': 'badge-orange',
    'Deployed': 'badge-green', 'Rejected': 'badge-red'
}

const SUFF_BADGE = {
    'Highly Sufficient': 'badge-green', 'Moderately Sufficient': 'badge-orange',
    'Not Sufficient': 'badge-red'
}
const SUFF_ICON = { 'Highly Sufficient': '🟢', 'Moderately Sufficient': '🟡', 'Not Sufficient': '🔴' }

const MAP_STATUS_COLORS = {
    'Deployed': '#10b981',
    'Approved': '#3b82f6',
    'Installation In Progress': '#f59e0b'
}

export default function AdminPanel() {
    const { getAuthHeaders } = useAuth()
    const [apps, setApps] = useState([])
    const [stats, setStats] = useState(null)
    const [total, setTotal] = useState(0)
    const [page, setPage] = useState(1)
    const [pages, setPages] = useState(1)
    const [filter, setFilter] = useState({ status: 'all', city: 'all', search: '' })
    const [loading, setLoading] = useState(true)
    const [editingId, setEditingId] = useState(null)
    const [deployedStations, setDeployedStations] = useState([])
    const [showMap, setShowMap] = useState(true)

    const fetchApps = (p = 1) => {
        const params = new URLSearchParams({ page: p, per_page: 15 })
        if (filter.status !== 'all') params.set('status', filter.status)
        if (filter.city !== 'all') params.set('city', filter.city)
        if (filter.search) params.set('search', filter.search)

        fetch(`${API}/applications?${params}`, { headers: getAuthHeaders() }).then(r => r.json()).then(d => {
            setApps(d.applications || [])
            setTotal(d.total || 0)
            setPages(d.pages || 1)
            setPage(p)
        })
    }

    const fetchStats = () => {
        fetch(`${API}/stats`, { headers: getAuthHeaders() }).then(r => r.json()).then(setStats)
    }

    const fetchDeployed = () => {
        fetch(`${API}/deployed`, { headers: getAuthHeaders() }).then(r => r.json()).then(setDeployedStations)
    }

    useEffect(() => {
        Promise.all([fetchApps(), fetchStats(), fetchDeployed()]).then(() => setLoading(false))
    }, [])

    useEffect(() => { fetchApps(1) }, [filter])

    const updateStatus = (appId, newStatus) => {
        fetch(`${API}/applications/${appId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify({ status: newStatus })
        }).then(r => r.json()).then(() => {
            setEditingId(null)
            fetchApps(page)
            fetchStats()
            fetchDeployed()
        })
    }

    if (loading) return <div className="loading"><div className="spinner" />Loading admin data...</div>

    const statusData = stats?.status_breakdown ? {
        labels: Object.keys(stats.status_breakdown),
        datasets: [{
            data: Object.values(stats.status_breakdown),
            backgroundColor: ['#3b82f6', '#f59e0b', '#8b5cf6', '#8b5cf6', '#10b981', '#f59e0b', '#10b981', '#ef4444']
        }]
    } : null

    const monthlyData = stats?.monthly_apps ? {
        labels: Object.keys(stats.monthly_apps).map(m =>
            ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m] || m),
        datasets: [{
            label: 'Applications',
            data: Object.values(stats.monthly_apps),
            backgroundColor: '#3b82f6',
            borderRadius: 6
        }]
    } : null

    const cities = stats?.city_breakdown ? Object.keys(stats.city_breakdown) : []

    return (
        <div className="page">
            <div className="page-header">
                <h1>⚙️ Admin Panel</h1>
                <p>Manage EV charging station applications and monitor system health</p>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon blue">📋</div>
                    <div className="stat-info"><h3>{stats?.total_applications || 0}</h3><p>Total Applications</p></div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon orange">⏳</div>
                    <div className="stat-info"><h3>{stats?.pending || 0}</h3><p>Pending Review</p></div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon green">✅</div>
                    <div className="stat-info"><h3>{stats?.approved || 0}</h3><p>Approved</p></div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon red">🚀</div>
                    <div className="stat-info"><h3>{stats?.deployed || 0}</h3><p>Deployed</p></div>
                </div>
            </div>

            <div className="grid-2">
                {statusData && (
                    <div className="card">
                        <div className="card-header"><span className="card-title">Status Breakdown</span></div>
                        <div className="chart-container">
                            <Doughnut data={statusData} options={{
                                responsive: true, maintainAspectRatio: false,
                                plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } } }
                            }} />
                        </div>
                    </div>
                )}
                {monthlyData && (
                    <div className="card">
                        <div className="card-header"><span className="card-title">Monthly Applications</span></div>
                        <div className="chart-container">
                            <Bar data={monthlyData} options={{
                                responsive: true, maintainAspectRatio: false,
                                plugins: { legend: { display: false } },
                                scales: {
                                    x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
                                    y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
                                }
                            }} />
                        </div>
                    </div>
                )}
            </div>

            {/* Deployed Stations Map */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
                <div className="card-header">
                    <span className="card-title">🗺️ Deployed & Approved Stations ({deployedStations.length})</span>
                    <button className="btn btn-sm btn-secondary" onClick={() => setShowMap(!showMap)}>
                        {showMap ? '▲ Hide Map' : '▼ Show Map'}
                    </button>
                </div>
                {showMap && (
                    <div className="map-container" style={{ height: '400px' }}>
                        <MapContainer
                            center={[15, 78]}
                            zoom={5}
                            style={{ height: '100%', width: '100%' }}
                        >
                            <TileLayer
                                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                                attribution='&copy; CARTO'
                            />
                            {deployedStations.map((s, i) => (
                                <CircleMarker
                                    key={i}
                                    center={[s.lat, s.lng]}
                                    radius={8}
                                    pathOptions={{
                                        color: MAP_STATUS_COLORS[s.status] || '#3b82f6',
                                        fillOpacity: 0.7
                                    }}
                                >
                                    <Popup>
                                        <strong>{s.applicant}</strong><br />
                                        📍 {s.locality}, {s.city}<br />
                                        🏢 {s.property_type}<br />
                                        ⚡ {s.requested_chargers} × {s.charger_type}<br />
                                        <span style={{ fontWeight: 600, color: MAP_STATUS_COLORS[s.status] }}>
                                            {s.status}
                                        </span>
                                    </Popup>
                                </CircleMarker>
                            ))}
                        </MapContainer>
                    </div>
                )}
                {showMap && deployedStations.length > 0 && (
                    <div style={{ display: 'flex', gap: '1.5rem', padding: '0.75rem 1rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        <span>🟢 Deployed</span>
                        <span>🔵 Approved</span>
                        <span>🟡 Installing</span>
                    </div>
                )}
            </div>

            {/* Applications Table */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
                <div className="card-header">
                    <span className="card-title">Applications ({total})</span>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <input className="form-control" placeholder="Search..." style={{ width: '200px' }}
                            value={filter.search} onChange={e => setFilter({ ...filter, search: e.target.value })} />
                        <select className="form-control" style={{ width: '160px' }}
                            value={filter.status} onChange={e => setFilter({ ...filter, status: e.target.value })}>
                            <option value="all">All Statuses</option>
                            {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                        {cities.length > 0 && (
                            <select className="form-control" style={{ width: '140px' }}
                                value={filter.city} onChange={e => setFilter({ ...filter, city: e.target.value })}>
                                <option value="all">All Cities</option>
                                {cities.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        )}
                    </div>
                </div>

                <div className="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>ID</th><th>Applicant</th><th>City</th><th>Property</th>
                                <th>Chargers</th><th>EV Sufficiency</th><th>Date</th><th>Status</th><th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {apps.map(a => (
                                <tr key={a.application_id}>
                                    <td style={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>{a.application_id}</td>
                                    <td style={{ fontWeight: 500 }}>{a.applicant}</td>
                                    <td>{a.city || a.locality || '—'}</td>
                                    <td>{a.property_type}</td>
                                    <td>{a.requested_chargers}</td>
                                    <td>
                                        {a.sufficiency_status ? (
                                            <span className={`badge ${SUFF_BADGE[a.sufficiency_status] || 'badge-blue'}`}>
                                                {SUFF_ICON[a.sufficiency_status] || '❓'} {a.sufficiency_status}
                                            </span>
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>—</span>
                                        )}
                                    </td>
                                    <td style={{ fontSize: '0.8rem' }}>{a.date}</td>
                                    <td>
                                        <span className={`badge ${STATUS_BADGE[a.status] || 'badge-blue'}`}>
                                            {a.status}
                                        </span>
                                    </td>
                                    <td>
                                        {editingId === a.application_id ? (
                                            <select className="form-control" style={{ width: '160px', padding: '0.3rem' }}
                                                defaultValue={a.status}
                                                onChange={e => updateStatus(a.application_id, e.target.value)}>
                                                {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
                                            </select>
                                        ) : (
                                            <button className="btn btn-sm btn-secondary"
                                                onClick={() => setEditingId(a.application_id)}>
                                                ✏️ Edit
                                            </button>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="pagination">
                    <button disabled={page <= 1} onClick={() => fetchApps(page - 1)}>← Prev</button>
                    <span className="current">Page {page} of {pages}</span>
                    <button disabled={page >= pages} onClick={() => fetchApps(page + 1)}>Next →</button>
                </div>
            </div>
        </div>
    )
}
