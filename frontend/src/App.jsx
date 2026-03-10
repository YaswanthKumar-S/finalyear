import { Routes, Route, NavLink, Navigate, useNavigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './AuthContext'
import ProtectedRoute from './ProtectedRoute'
import DemandDashboard from './pages/DemandDashboard'
import LocationPlanner from './pages/LocationPlanner'
import RouteFinder from './pages/RouteFinder'
import ClientApplication from './pages/ClientApplication'
import Chatbot from './pages/Chatbot'
import AdminPanel from './pages/AdminPanel'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'

function Navbar() {
    const { isAuthenticated, isAdmin, isClient, user, logout } = useAuth()
    const navigate = useNavigate()

    const handleLogout = () => {
        logout()
        navigate('/login')
    }

    return (
        <nav className="navbar">
            <NavLink to={isAdmin ? '/' : '/route'} className="navbar-brand">
                ⚡ <span>EV Station Manager</span>
            </NavLink>
            <div className="navbar-links">
                {/* Admin-only links */}
                {isAdmin && (
                    <>
                        <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
                            📊 Demand
                        </NavLink>
                        <NavLink to="/location" className={({ isActive }) => isActive ? 'active' : ''}>
                            📍 Location
                        </NavLink>
                    </>
                )}

                {/* Public links */}
                <NavLink to="/route" className={({ isActive }) => isActive ? 'active' : ''}>
                    🗺️ Route
                </NavLink>

                {/* Client + Admin links */}
                {(isAdmin || isClient) && (
                    <NavLink to="/apply" className={({ isActive }) => isActive ? 'active' : ''}>
                        📝 Apply
                    </NavLink>
                )}

                <NavLink to="/chat" className={({ isActive }) => isActive ? 'active' : ''}>
                    🤖 Chat
                </NavLink>

                {/* Admin only */}
                {isAdmin && (
                    <NavLink to="/admin" className={({ isActive }) => isActive ? 'active' : ''}>
                        ⚙️ Admin
                    </NavLink>
                )}

                {/* Auth actions */}
                {isAuthenticated ? (
                    <div className="navbar-user">
                        <span className="navbar-user-badge">
                            {user?.role === 'admin' ? '🛡️' : '👤'} {user?.name?.split(' ')[0]}
                        </span>
                        <button className="btn btn-sm navbar-logout" onClick={handleLogout}>
                            Logout
                        </button>
                    </div>
                ) : (
                    <NavLink to="/login" className={({ isActive }) => `nav-login ${isActive ? 'active' : ''}`}>
                        🔐 Login
                    </NavLink>
                )}
            </div>
        </nav>
    )
}

function AppRoutes() {
    const { isAuthenticated, loading } = useAuth()

    if (loading) {
        return <div className="loading"><div className="spinner" />Loading...</div>
    }

    return (
        <Routes>
            {/* Public routes */}
            <Route path="/login" element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />} />
            <Route path="/register" element={isAuthenticated ? <Navigate to="/" replace /> : <RegisterPage />} />
            <Route path="/route" element={<><Navbar /><RouteFinder /></>} />
            <Route path="/chat" element={<><Navbar /><Chatbot /></>} />

            {/* Admin routes */}
            <Route path="/" element={
                <ProtectedRoute role="admin">
                    <Navbar /><DemandDashboard />
                </ProtectedRoute>
            } />
            <Route path="/location" element={
                <ProtectedRoute role="admin">
                    <Navbar /><LocationPlanner />
                </ProtectedRoute>
            } />
            <Route path="/admin" element={
                <ProtectedRoute role="admin">
                    <Navbar /><AdminPanel />
                </ProtectedRoute>
            } />

            {/* Client routes */}
            <Route path="/apply" element={
                <ProtectedRoute role="client">
                    <Navbar /><ClientApplication />
                </ProtectedRoute>
            } />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/route" replace />} />
        </Routes>
    )
}

export default function App() {
    return (
        <AuthProvider>
            <AppRoutes />
        </AuthProvider>
    )
}
