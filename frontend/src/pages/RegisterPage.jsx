import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../AuthContext'

export default function RegisterPage() {
    const { register } = useAuth()
    const navigate = useNavigate()
    const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' })
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const set = (k, v) => setForm({ ...form, [k]: v })

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')

        if (!form.name.trim()) return setError('Name is required')
        if (!form.email.includes('@')) return setError('Valid email is required')
        if (form.password.length < 6) return setError('Password must be at least 6 characters')
        if (form.password !== form.confirm) return setError('Passwords do not match')

        setLoading(true)
        try {
            await register(form.name, form.email, form.password)
            navigate('/apply')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="auth-page">
            <div className="auth-bg">
                <div className="auth-orb auth-orb-1" />
                <div className="auth-orb auth-orb-2" />
                <div className="auth-orb auth-orb-3" />
            </div>
            <div className="auth-card">
                <div className="auth-logo">
                    <span className="auth-logo-icon">⚡</span>
                    <h1>Create Client Account</h1>
                    <p>Register to apply for EV charging stations</p>
                </div>

                <form onSubmit={handleSubmit} className="auth-form">
                    {error && (
                        <div className="auth-error">
                            <span>⚠️</span> {error}
                        </div>
                    )}

                    <div className="auth-field">
                        <label>Full Name</label>
                        <input
                            type="text"
                            className="form-control"
                            placeholder="Rajesh Kumar"
                            value={form.name}
                            onChange={e => set('name', e.target.value)}
                            required
                            autoFocus
                        />
                    </div>

                    <div className="auth-field">
                        <label>Email Address</label>
                        <input
                            type="email"
                            className="form-control"
                            placeholder="you@example.com"
                            value={form.email}
                            onChange={e => set('email', e.target.value)}
                            required
                        />
                    </div>

                    <div className="auth-field">
                        <label>Password</label>
                        <input
                            type="password"
                            className="form-control"
                            placeholder="Min. 6 characters"
                            value={form.password}
                            onChange={e => set('password', e.target.value)}
                            required
                        />
                    </div>

                    <div className="auth-field">
                        <label>Confirm Password</label>
                        <input
                            type="password"
                            className="form-control"
                            placeholder="Re-enter password"
                            value={form.confirm}
                            onChange={e => set('confirm', e.target.value)}
                            required
                        />
                    </div>

                    <button type="submit" className="btn btn-primary auth-submit" disabled={loading}>
                        {loading ? '⏳ Creating account...' : '🚀 Create Account'}
                    </button>
                </form>

                <div className="auth-footer">
                    <p>Already have an account? <Link to="/login">Sign In</Link></p>
                </div>
            </div>
        </div>
    )
}
