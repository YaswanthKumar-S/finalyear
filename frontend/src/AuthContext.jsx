import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext(null)

export function useAuth() {
    const ctx = useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [token, setToken] = useState(localStorage.getItem('ev_token'))
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (token) {
            fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            })
                .then(r => {
                    if (!r.ok) throw new Error('Invalid token')
                    return r.json()
                })
                .then(u => { setUser(u); setLoading(false) })
                .catch(() => { logout(); setLoading(false) })
        } else {
            setLoading(false)
        }
    }, [])

    const login = async (email, password) => {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.error || 'Login failed')
        localStorage.setItem('ev_token', data.token)
        setToken(data.token)
        setUser(data.user)
        return data.user
    }

    const register = async (name, email, password) => {
        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, password })
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.error || 'Registration failed')
        localStorage.setItem('ev_token', data.token)
        setToken(data.token)
        setUser(data.user)
        return data.user
    }

    const logout = () => {
        localStorage.removeItem('ev_token')
        setToken(null)
        setUser(null)
    }

    const getAuthHeaders = () => {
        if (!token) return {}
        return { 'Authorization': `Bearer ${token}` }
    }

    return (
        <AuthContext.Provider value={{
            user, token, loading,
            isAuthenticated: !!user,
            isAdmin: user?.role === 'admin',
            isClient: user?.role === 'client',
            login, register, logout, getAuthHeaders
        }}>
            {children}
        </AuthContext.Provider>
    )
}
