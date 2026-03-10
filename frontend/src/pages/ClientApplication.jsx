import { useState } from 'react'
import { useAuth } from '../AuthContext'

const API = '/api/client'

const PROPERTY_TYPES = ['Shopping Mall', 'Petrol Bunk', 'Commercial Complex', 'Hotel/Restaurant',
    'Apartment Complex', 'IT Park', 'Hospital', 'Standalone Parking', 'Highway Rest Area',
    'Supermarket', 'Farm House', 'Village Panchayat', 'Educational Institution']

const CHARGER_TYPES = ['AC Level 2 (7.4kW)', 'AC Level 2 (22kW)', 'DC Fast (50kW)',
    'DC Fast (100kW)', 'DC Ultra-Fast (150kW)']

const CITIES = ['Chennai', 'Coimbatore', 'Madurai', 'Trichy', 'Salem', 'Bangalore', 'Hyderabad', 'Mumbai', 'Delhi', 'Pune']
const AREA_TYPES = ['Metro', 'Suburban', 'Village']
const HOURS_OPTIONS = ['24/7', '6AM-10PM', '8AM-8PM', '8AM-6PM']

const SUIT_BADGE = { 'Highly Suitable': 'badge-green', 'Suitable': 'badge-blue', 'Moderate': 'badge-orange', 'Low': 'badge-red' }
const SUFF_BADGE = { 'Highly Sufficient': 'badge-green', 'Moderately Sufficient': 'badge-orange', 'Not Sufficient': 'badge-red' }
const SUFF_ICON = { 'Highly Sufficient': '\u25CF', 'Moderately Sufficient': '\u25CF', 'Not Sufficient': '\u25CF' }
const SUFF_COLOR = { 'Highly Sufficient': '#22c55e', 'Moderately Sufficient': '#eab308', 'Not Sufficient': '#ef4444' }

function Field({ label, error, children }) {
    return (
        <div className="form-group">
            <label>{label}</label>
            {children}
            {error && <span style={{ color: 'var(--accent-red)', fontSize: '0.75rem' }}>{error}</span>}
        </div>
    )
}

export default function ClientApplication() {
    const { getAuthHeaders } = useAuth()
    const [form, setForm] = useState({
        applicant: '', email: '', phone: '', company: '',
        property_type: 'Shopping Mall', city: 'Chennai', locality: '',
        area_type: 'Metro', latitude: '', longitude: '',
        area_sqft: '', parking_spaces: '', requested_chargers: '1',
        charger_type: 'DC Fast (50kW)', power_kw: '50',
        has_solar: false, hours: '24/7'
    })
    const [result, setResult] = useState(null)
    const [errors, setErrors] = useState({})
    const [submitting, setSubmitting] = useState(false)
    const [trackId, setTrackId] = useState('')
    const [tracked, setTracked] = useState(null)

    const set = (k, v) => setForm(prev => ({ ...prev, [k]: v }))

    const validate = () => {
        const e = {}
        if (!form.applicant.trim()) e.applicant = 'Name is required'
        if (!form.email.includes('@')) e.email = 'Valid email required'
        if (form.phone.replace(/\D/g, '').length < 10) e.phone = 'Valid phone required'
        if (!form.locality.trim()) e.locality = 'Locality is required'
        if (!form.latitude || !form.longitude) e.location = 'Coordinates required'
        if (!form.area_sqft || +form.area_sqft < 100) e.area_sqft = 'Min 100 sq ft'
        if (!form.parking_spaces || +form.parking_spaces < 1) e.parking_spaces = 'Min 1 space'
        setErrors(e)
        return Object.keys(e).length === 0
    }

    const submit = () => {
        if (!validate()) return
        setSubmitting(true)
        fetch(`${API}/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify(form)
        }).then(r => r.json()).then(d => {
            setResult(d)
            setSubmitting(false)
        }).catch(() => setSubmitting(false))
    }

    const track = () => {
        if (!trackId.trim()) return
        fetch(`${API}/track/${trackId.trim()}`).then(r => r.json()).then(setTracked)
    }

    const useMyLocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(pos => {
                setForm(prev => ({
                    ...prev,
                    latitude: pos.coords.latitude.toFixed(6),
                    longitude: pos.coords.longitude.toFixed(6)
                }))
            })
        }
    }

    if (result?.success) {
        const la = result.location_analysis || {}
        return (
            <div className="page">
                <div style={{ maxWidth: 720, margin: '3rem auto', textAlign: 'center' }}>
                    <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>✅</div>
                    <h1 style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>Application Submitted!</h1>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>
                        Your application has been received and is under review.
                    </p>

                    {/* Application Info */}
                    <div className="card" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">📋 Application Details</span></div>
                        <div style={{ display: 'grid', gap: '0.75rem' }}>
                            <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Application ID</span>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'monospace' }}>
                                    {result.application_id}
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                                <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Applicant</span>
                                    <div>{form.applicant}</div>
                                </div>
                                <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Location</span>
                                    <div>{form.locality}, {form.city}</div>
                                </div>
                            </div>
                            <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Status</span>
                                <div><span className="badge badge-blue">{result.status}</span></div>
                            </div>
                        </div>
                    </div>

                    {/* Location & Demand Analysis */}
                    <div className="card" style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">📊 Location & Demand Analysis</span></div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)' }}>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Suitability</div>
                                <span className={`badge ${SUIT_BADGE[la.suitability] || 'badge-blue'}`}>{la.suitability || 'N/A'}</span>
                            </div>
                            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)' }}>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Optimal Score</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-green)' }}>{la.optimal_score || 0}</div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>out of 100</div>
                            </div>
                            <div style={{ textAlign: 'center', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)' }}>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Avg Demand</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#3b82f6' }}>{la.demand_estimate || 0}</div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>vehicles/hour</div>
                            </div>
                        </div>

                        {/* EV Sufficiency */}
                        <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', marginBottom: '1rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                                <span style={{ fontWeight: 600 }}>EV Charging Sufficiency</span>
                                <span className={`badge ${SUFF_BADGE[la.sufficiency_status] || 'badge-blue'}`}>
                                    {SUFF_ICON[la.sufficiency_status] || '❓'} {la.sufficiency_status || 'Unknown'}
                                </span>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '0.75rem' }}>
                                <div><span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>EVs in Area</span>
                                    <div style={{ fontWeight: 600 }}>~{la.ev_count_area || 0}</div>
                                </div>
                                <div><span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Nearby Stations</span>
                                    <div style={{ fontWeight: 600 }}>{la.station_count_area || 0}</div>
                                </div>
                                <div><span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Within</span>
                                    <div style={{ fontWeight: 600 }}>15 km</div>
                                </div>
                            </div>
                            {la.recommendation && (
                                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic', padding: '0.5rem', background: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)' }}>
                                    💡 {la.recommendation}
                                </div>
                            )}
                        </div>
                    </div>

                    <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                        Save your Application ID to track status. You will receive updates at {form.email}.
                    </p>
                    <button className="btn btn-primary" style={{ marginTop: '1.5rem' }}
                        onClick={() => { setResult(null); setForm(prev => ({ ...prev, applicant: '', email: '', phone: '', company: '', locality: '', area_sqft: '', parking_spaces: '', latitude: '', longitude: '' })) }}>
                        📝 Submit Another
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="page">
            <div className="page-header">
                <h1>📝 Apply for EV Charger Installation</h1>
                <p>Submit your application to install EV charging stations at your property</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
                <div>
                    {/* Personal Info */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">👤 Applicant Details</span></div>
                        <div className="grid-2">
                            <Field label="Full Name *" error={errors.applicant}>
                                <input className="form-control" placeholder="Rajesh Kumar" value={form.applicant}
                                    onChange={e => set('applicant', e.target.value)} />
                            </Field>
                            <Field label="Company (Optional)">
                                <input className="form-control" placeholder="Kumar Enterprises" value={form.company}
                                    onChange={e => set('company', e.target.value)} />
                            </Field>
                        </div>
                        <div className="grid-2">
                            <Field label="Email *" error={errors.email}>
                                <input type="email" className="form-control" placeholder="rajesh@gmail.com" value={form.email}
                                    onChange={e => set('email', e.target.value)} />
                            </Field>
                            <Field label="Phone *" error={errors.phone}>
                                <input className="form-control" placeholder="+91 98765 43210" value={form.phone}
                                    onChange={e => set('phone', e.target.value)} />
                            </Field>
                        </div>
                    </div>

                    {/* Property Info */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">🏢 Property Details</span></div>
                        <div className="grid-2">
                            <Field label="Property Type *">
                                <select className="form-control" value={form.property_type}
                                    onChange={e => set('property_type', e.target.value)}>
                                    {PROPERTY_TYPES.map(p => <option key={p} value={p}>{p}</option>)}
                                </select>
                            </Field>
                            <Field label="City *">
                                <select className="form-control" value={form.city}
                                    onChange={e => set('city', e.target.value)}>
                                    {CITIES.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </Field>
                        </div>
                        <div className="grid-2">
                            <Field label="Locality / Area *" error={errors.locality}>
                                <input className="form-control" placeholder="T. Nagar, Anna Nagar..." value={form.locality}
                                    onChange={e => set('locality', e.target.value)} />
                            </Field>
                            <Field label="Area Type">
                                <select className="form-control" value={form.area_type}
                                    onChange={e => set('area_type', e.target.value)}>
                                    {AREA_TYPES.map(a => <option key={a} value={a}>{a}</option>)}
                                </select>
                            </Field>
                        </div>
                        <div className="grid-3">
                            <Field label="Latitude *" error={errors.location}>
                                <input type="number" step="0.0001" className="form-control" placeholder="13.0827" value={form.latitude}
                                    onChange={e => set('latitude', e.target.value)} />
                            </Field>
                            <Field label="Longitude *">
                                <input type="number" step="0.0001" className="form-control" placeholder="80.2707" value={form.longitude}
                                    onChange={e => set('longitude', e.target.value)} />
                            </Field>
                            <div className="form-group" style={{ display: 'flex', alignItems: 'flex-end' }}>
                                <button className="btn btn-secondary" onClick={useMyLocation} style={{ width: '100%', justifyContent: 'center' }}>
                                    📱 Use GPS
                                </button>
                            </div>
                        </div>
                        <div className="grid-2">
                            <Field label="Area (sq ft) *" error={errors.area_sqft}>
                                <input type="number" className="form-control" placeholder="5000" value={form.area_sqft}
                                    onChange={e => set('area_sqft', e.target.value)} />
                            </Field>
                            <Field label="Parking Spaces *" error={errors.parking_spaces}>
                                <input type="number" className="form-control" placeholder="20" value={form.parking_spaces}
                                    onChange={e => set('parking_spaces', e.target.value)} />
                            </Field>
                        </div>
                    </div>

                    {/* Charger Specs */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">⚡ Charger Specifications</span></div>
                        <div className="grid-3">
                            <Field label="Number of Chargers *">
                                <input type="number" min="1" max="50" className="form-control" value={form.requested_chargers}
                                    onChange={e => set('requested_chargers', e.target.value)} />
                            </Field>
                            <Field label="Charger Type">
                                <select className="form-control" value={form.charger_type}
                                    onChange={e => set('charger_type', e.target.value)}>
                                    {CHARGER_TYPES.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </Field>
                            <Field label="Power (kW)">
                                <select className="form-control" value={form.power_kw}
                                    onChange={e => set('power_kw', e.target.value)}>
                                    {[7, 22, 50, 100, 150, 200, 300].map(p => <option key={p} value={p}>{p} kW</option>)}
                                </select>
                            </Field>
                        </div>
                        <div className="grid-2">
                            <Field label="Operating Hours">
                                <select className="form-control" value={form.hours}
                                    onChange={e => set('hours', e.target.value)}>
                                    {HOURS_OPTIONS.map(h => <option key={h} value={h}>{h}</option>)}
                                </select>
                            </Field>
                            <Field label="Solar Power Available?">
                                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', textTransform: 'none', letterSpacing: 0 }}>
                                        <input type="radio" name="solar" checked={form.has_solar} onChange={() => set('has_solar', true)} /> Yes
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', textTransform: 'none', letterSpacing: 0 }}>
                                        <input type="radio" name="solar" checked={!form.has_solar} onChange={() => set('has_solar', false)} /> No
                                    </label>
                                </div>
                            </Field>
                        </div>
                    </div>

                    <button className="btn btn-primary" onClick={submit} disabled={submitting}
                        style={{ width: '100%', justifyContent: 'center', padding: '0.875rem', fontSize: '1rem' }}>
                        {submitting ? '⏳ Submitting...' : '🚀 Submit Application'}
                    </button>
                </div>

                {/* Sidebar */}
                <div>
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">📋 Requirements</span></div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                            <div>✅ Min 500 sq ft available area</div>
                            <div>✅ 3-phase power connection</div>
                            <div>✅ At least 5 parking spaces</div>
                            <div>✅ Property ownership proof</div>
                            <div>✅ NOC from local authority</div>
                            <div>✅ Electrical safety certificate</div>
                        </div>
                    </div>

                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><span className="card-title">📊 Process</span></div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            {['Applied', 'Under Review', 'Site Visit', 'Approval', 'Installation', 'Deployed'].map((step, i) => (
                                <div key={i} style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.75rem', alignItems: 'center' }}>
                                    <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'var(--accent-green-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-green)', flexShrink: 0 }}>
                                        {i + 1}
                                    </div>
                                    <span>{step}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header"><span className="card-title">Track Application</span></div>
                        <Field label="Application ID">
                            <input className="form-control" placeholder="EV-APP-2024-XXXX" value={trackId}
                                onChange={e => setTrackId(e.target.value)} />
                        </Field>
                        <button className="btn btn-secondary" onClick={track} style={{ width: '100%', justifyContent: 'center' }}>
                            Track Status
                        </button>
                        {tracked && (
                            <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)' }}>
                                {tracked.error ? (
                                    <p style={{ color: 'var(--accent-red)' }}>{tracked.error}</p>
                                ) : (
                                    <>
                                        <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>{tracked.applicant}</div>
                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                                            {tracked.locality}, {tracked.city}
                                        </div>
                                        <span className={`badge ${tracked.status === 'Deployed' || tracked.status === 'Approved' ? 'badge-green' :
                                            tracked.status === 'Rejected' ? 'badge-red' : 'badge-orange'
                                            }`}>{tracked.status}</span>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
