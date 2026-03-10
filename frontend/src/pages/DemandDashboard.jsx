import { useState, useEffect } from 'react'
import { useAuth } from '../AuthContext'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import {
    Chart as ChartJS, CategoryScale, LinearScale, PointElement,
    LineElement, BarElement, ArcElement, Title, Tooltip, Legend, Filler
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement,
    BarElement, ArcElement, Title, Tooltip, Legend, Filler)

const API = '/api/demand'

export default function DemandDashboard() {
    const { getAuthHeaders } = useAuth()
    const [analytics, setAnalytics] = useState(null)
    const [stations, setStations] = useState([])
    const [prediction, setPrediction] = useState(null)
    const [form, setForm] = useState({
        station_id: '', hour: 12, day_of_week: 0, month: 1,
        temperature: 30, is_raining: 0, is_holiday: 0
    })
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch(`${API}/analytics`, { headers: getAuthHeaders() }).then(r => r.json()),
            fetch(`${API}/stations`, { headers: getAuthHeaders() }).then(r => r.json())
        ]).then(([anal, stn]) => {
            setAnalytics(anal)
            setStations(stn)
            if (stn.length > 0) setForm(f => ({ ...f, station_id: stn[0].station_id }))
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    const predict = () => {
        fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify(form)
        }).then(r => r.json()).then(setPrediction)
    }

    if (loading) return <div className="loading"><div className="spinner" />Loading demand data...</div>

    const hours = Object.keys(analytics?.hourly || {})
    const hourlyData = {
        labels: hours.map(h => `${h}:00`),
        datasets: [{
            label: 'Avg Demand',
            data: Object.values(analytics?.hourly || {}),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16,185,129,0.1)',
            fill: true, tension: 0.4, pointRadius: 3
        }]
    }

    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    const dailyData = {
        labels: days,
        datasets: [{
            label: 'Avg Demand',
            data: days.map((_, i) => analytics?.daily?.[i] || 0),
            backgroundColor: days.map((_, i) => i >= 5 ? '#f59e0b' : '#3b82f6'),
            borderRadius: 6
        }]
    }

    const monthlyData = {
        labels: Array.from({ length: 12 }, (_, i) => ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i]),
        datasets: [{
            label: 'Monthly Avg',
            data: Array.from({ length: 12 }, (_, i) => analytics?.monthly?.[i + 1] || 0),
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139,92,246,0.1)',
            fill: true, tension: 0.4
        }]
    }

    const cityData = analytics?.city_demand ? {
        labels: Object.keys(analytics.city_demand),
        datasets: [{
            data: Object.values(analytics.city_demand),
            backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16']
        }]
    } : null

    const chartOpts = {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
            y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
        }
    }

    const s = analytics?.stats || {}

    return (
        <div className="page">
            <div className="page-header">
                <h1>📊 Demand Prediction Dashboard</h1>
                <p>Real-time EV charging demand analytics and predictions across India</p>
            </div>

            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon green">⚡</div>
                    <div className="stat-info">
                        <h3>{s.avg_demand?.toFixed(1) || '—'}</h3>
                        <p>Avg Demand / Hour</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon blue">🔋</div>
                    <div className="stat-info">
                        <h3>{s.avg_energy_kwh?.toFixed(0) || '—'} kWh</h3>
                        <p>Avg Energy / Hour</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon orange">📈</div>
                    <div className="stat-info">
                        <h3>{s.avg_utilization?.toFixed(0) || '—'}%</h3>
                        <p>Avg Utilization</p>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon red">🏢</div>
                    <div className="stat-info">
                        <h3>{stations.length}</h3>
                        <p>Total Stations</p>
                    </div>
                </div>
            </div>

            <div className="grid-2">
                <div className="card">
                    <div className="card-header"><span className="card-title">Hourly Demand Pattern</span></div>
                    <div className="chart-container"><Line data={hourlyData} options={chartOpts} /></div>
                </div>
                <div className="card">
                    <div className="card-header"><span className="card-title">Daily Demand Pattern</span></div>
                    <div className="chart-container"><Bar data={dailyData} options={chartOpts} /></div>
                </div>
            </div>

            <div className="grid-2">
                <div className="card">
                    <div className="card-header"><span className="card-title">Monthly Trend</span></div>
                    <div className="chart-container"><Line data={monthlyData} options={chartOpts} /></div>
                </div>
                {cityData && (
                    <div className="card">
                        <div className="card-header"><span className="card-title">Demand by City</span></div>
                        <div className="chart-container">
                            <Doughnut data={cityData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } } } }} />
                        </div>
                    </div>
                )}
            </div>

            <div className="grid-2" style={{ marginTop: '1.5rem' }}>
                <div className="card">
                    <div className="card-header"><span className="card-title">🔮 Predict Demand</span></div>
                    <div className="form-group">
                        <label>Station</label>
                        <select className="form-control" value={form.station_id}
                            onChange={e => setForm({ ...form, station_id: e.target.value })}>
                            {stations.map(s => (
                                <option key={s.station_id} value={s.station_id}>
                                    {s.station_name || s.name} ({s.city || 'Chennai'})
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="grid-3">
                        <div className="form-group">
                            <label>Hour (0-23)</label>
                            <input type="number" className="form-control" min="0" max="23"
                                value={form.hour} onChange={e => setForm({ ...form, hour: +e.target.value })} />
                        </div>
                        <div className="form-group">
                            <label>Day (0=Mon)</label>
                            <select className="form-control" value={form.day_of_week}
                                onChange={e => setForm({ ...form, day_of_week: +e.target.value })}>
                                {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((d, i) =>
                                    <option key={i} value={i}>{d}</option>)}
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Month</label>
                            <select className="form-control" value={form.month}
                                onChange={e => setForm({ ...form, month: +e.target.value })}>
                                {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map((m, i) =>
                                    <option key={i} value={i + 1}>{m}</option>)}
                            </select>
                        </div>
                    </div>
                    <div className="grid-3">
                        <div className="form-group">
                            <label>Temperature °C</label>
                            <input type="number" className="form-control" value={form.temperature}
                                onChange={e => setForm({ ...form, temperature: +e.target.value })} />
                        </div>
                        <div className="form-group">
                            <label>Raining?</label>
                            <select className="form-control" value={form.is_raining}
                                onChange={e => setForm({ ...form, is_raining: +e.target.value })}>
                                <option value={0}>No</option><option value={1}>Yes</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Holiday?</label>
                            <select className="form-control" value={form.is_holiday}
                                onChange={e => setForm({ ...form, is_holiday: +e.target.value })}>
                                <option value={0}>No</option><option value={1}>Yes</option>
                            </select>
                        </div>
                    </div>
                    <button className="btn btn-primary" onClick={predict} style={{ width: '100%', justifyContent: 'center' }}>
                        ⚡ Predict Demand
                    </button>
                </div>

                <div className="card">
                    <div className="card-header"><span className="card-title">Prediction Result</span></div>
                    {prediction ? (
                        <div className="prediction-result">
                            <div className="score">{prediction.predicted_demand}</div>
                            <p style={{ fontSize: '1.1rem', marginTop: '0.5rem' }}>vehicles expected</p>
                            <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
                                <div><div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{prediction.utilization_pct}%</div><p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Utilization</p></div>
                                <div><div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{prediction.max_capacity}</div><p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Max Capacity</p></div>
                            </div>
                            <p className="confidence" style={{ marginTop: '1rem' }}>
                                Model: <span className="badge badge-green">{prediction.model}</span>
                            </p>
                        </div>
                    ) : (
                        <div className="prediction-result" style={{ opacity: 0.5 }}>
                            <p style={{ fontSize: '1.2rem' }}>Select parameters and click predict</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
