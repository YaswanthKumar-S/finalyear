import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../AuthContext'

export default function LoginPage() {
    const { login } = useAuth()
    const navigate = useNavigate()
    const [tab, setTab] = useState('admin')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setLoading(true)
        try {
            const user = await login(email, password)
            if (user.role === 'admin') {
                navigate('/')
            } else {
                navigate('/apply')
            }
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
                    <h1>EV Station Manager</h1>
                    <p>Sign in to your account</p>
                </div>

                <div className="auth-tabs">
                    <button
                        className={`auth-tab ${tab === 'admin' ? 'active' : ''}`}
                        onClick={() => { setTab('admin'); setEmail(''); setPassword(''); setError('') }}
                    >
                        🛡️ Admin
                    </button>
                    <button
                        className={`auth-tab ${tab === 'client' ? 'active' : ''}`}
                        onClick={() => { setTab('client'); setEmail(''); setPassword(''); setError('') }}
                    >
                        👤 Client
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="auth-form">
                    {error && (
                        <div className="auth-error">
                            <span>⚠️</span> {error}
                        </div>
                    )}

                    <div className="auth-field">
                        <label>Email Address</label>
                        <input
                            type="email"
                            className="form-control"
                            placeholder={tab === 'admin' ? 'admin@evstation.com' : 'you@example.com'}
                            value={email}
                            onChange={e => setEmail(e.target.value)}
                            required
                            autoFocus
                        />
                    </div>

                    <div className="auth-field">
                        <label>Password</label>
                        <input
                            type="password"
                            className="form-control"
                            placeholder="••••••••"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    <button type="submit" className="btn btn-primary auth-submit" disabled={loading}>
                        {loading ? '⏳ Signing in...' : '🔐 Sign In'}
                    </button>
                </form>

                {tab === 'client' && (
                    <div className="auth-footer">
                        <p>Don't have an account? <Link to="/register">Register as Client</Link></p>
                    </div>
                )}

                <div className="auth-public-links">
                    <p>Or continue without login:</p>
                    <div className="auth-public-btns">
                        <Link to="/route" className="btn btn-secondary">🗺️ Route Finder</Link>
                        <Link to="/chat" className="btn btn-secondary">🤖 Chatbot</Link>
                    </div>
                </div>
            </div>
        </div>
    )
}
