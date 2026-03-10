import { Navigate } from 'react-router-dom'
import { useAuth } from './AuthContext'

export default function ProtectedRoute({ role, children }) {
    const { isAuthenticated, user, loading } = useAuth()

    if (loading) {
        return (
            <div className="loading">
                <div className="spinner" />
                Checking authentication...
            </div>
        )
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />
    }

    // Role check: admin can access everything, client can access client + public
    if (role === 'admin' && user?.role !== 'admin') {
        return (
            <div className="page">
                <div className="access-denied">
                    <div className="access-denied-icon">🔒</div>
                    <h1>Access Denied</h1>
                    <p>You don't have permission to access this page.</p>
                    <p className="access-denied-hint">This page requires <strong>Admin</strong> privileges.</p>
                    <a href="/route" className="btn btn-primary">← Go to Route Finder</a>
                </div>
            </div>
        )
    }

    if (role === 'client' && !['client', 'admin'].includes(user?.role)) {
        return (
            <div className="page">
                <div className="access-denied">
                    <div className="access-denied-icon">🔒</div>
                    <h1>Access Denied</h1>
                    <p>Please log in as a client to access this page.</p>
                    <a href="/login" className="btn btn-primary">← Go to Login</a>
                </div>
            </div>
        )
    }

    return children
}
