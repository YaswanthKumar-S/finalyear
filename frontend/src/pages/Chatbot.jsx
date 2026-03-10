import { useState, useEffect, useRef } from 'react'
import config from '../config'

const API = '/api/chatbot'

export default function Chatbot() {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hi! 👋 I\'m your EV charging assistant. Ask me anything about electric vehicles, charging stations, battery maintenance, or EV policies in India!' }
    ])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const [apiKey, setApiKey] = useState(localStorage.getItem('groq_api_key') || config.CHAT_GROQ_API_KEY || '')
    const [showKeyInput, setShowKeyInput] = useState(!localStorage.getItem('groq_api_key') && !config.CHAT_GROQ_API_KEY)
    const [suggestions, setSuggestions] = useState([])
    const [sessionId] = useState(() => 'sess_' + Math.random().toString(36).slice(2, 10))
    const chatEndRef = useRef(null)
    const inputRef = useRef(null)

    useEffect(() => {
        fetch(`${API}/suggestions`).then(r => r.json()).then(setSuggestions).catch(() => { })
    }, [])

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const saveKey = () => {
        if (apiKey.trim()) {
            localStorage.setItem('groq_api_key', apiKey.trim())
            setShowKeyInput(false)
        }
    }

    const sendMessage = (text) => {
        const msg = text || input.trim()
        if (!msg || loading) return
        if (!apiKey) { setShowKeyInput(true); return }

        setMessages(prev => [...prev, { role: 'user', content: msg }])
        setInput('')
        setLoading(true)

        fetch(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, api_key: apiKey, session_id: sessionId })
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    setMessages(prev => [...prev, { role: 'error', content: data.error }])
                    if (data.error.includes('Invalid API key')) setShowKeyInput(true)
                } else {
                    setMessages(prev => [...prev, { role: 'assistant', content: data.reply }])
                }
                setLoading(false)
            })
            .catch(() => {
                setMessages(prev => [...prev, { role: 'error', content: 'Network error. Please try again.' }])
                setLoading(false)
            })
    }

    const clearChat = () => {
        fetch(`${API}/clear`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        })
        setMessages([
            { role: 'assistant', content: 'Chat cleared! Ask me anything about EVs. ⚡' }
        ])
    }

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    return (
        <div className="page" style={{ maxWidth: '900px' }}>
            <div className="page-header">
                <h1>🤖 EV Charging Assistant</h1>
                <p>Powered by ChatGroq (Llama 3.3 70B) — Ask anything about EVs</p>
            </div>

            {/* API Key input */}
            {showKeyInput && (
                <div className="card" style={{ marginBottom: '1.5rem', borderColor: 'var(--accent-orange)' }}>
                    <div className="card-header"><span className="card-title">🔑 Enter Groq API Key</span></div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                        Get your free API key at{' '}
                        <a href="https://console.groq.com" target="_blank" rel="noreferrer"
                            style={{ color: 'var(--accent-green)' }}>console.groq.com</a>
                    </p>
                    <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <input
                            type="password"
                            className="form-control"
                            placeholder="gsk_xxxxxxxxxxxx"
                            value={apiKey}
                            onChange={e => setApiKey(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && saveKey()}
                            style={{ flex: 1 }}
                        />
                        <button className="btn btn-primary" onClick={saveKey}>Save</button>
                    </div>
                </div>
            )}

            {/* Chat container */}
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                {/* Chat header */}
                <div style={{
                    padding: '0.75rem 1.25rem', borderBottom: '1px solid var(--border)',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    background: 'var(--bg-secondary)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{
                            width: 8, height: 8, borderRadius: '50%',
                            background: apiKey ? 'var(--accent-green)' : 'var(--accent-red)',
                            boxShadow: apiKey ? '0 0 8px var(--accent-green)' : 'none'
                        }} />
                        <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>
                            {apiKey ? 'Connected to Groq' : 'API key required'}
                        </span>
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button className="btn btn-sm btn-secondary" onClick={() => setShowKeyInput(!showKeyInput)}>
                            🔑
                        </button>
                        <button className="btn btn-sm btn-secondary" onClick={clearChat}>
                            🗑️ Clear
                        </button>
                    </div>
                </div>

                {/* Messages */}
                <div style={{
                    height: '480px', overflowY: 'auto', padding: '1.25rem',
                    display: 'flex', flexDirection: 'column', gap: '1rem'
                }}>
                    {messages.map((msg, i) => (
                        <div key={i} style={{
                            display: 'flex',
                            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                        }}>
                            <div style={{
                                maxWidth: '80%',
                                padding: '0.75rem 1rem',
                                borderRadius: msg.role === 'user'
                                    ? '16px 16px 4px 16px'
                                    : '16px 16px 16px 4px',
                                background: msg.role === 'user'
                                    ? 'linear-gradient(135deg, var(--accent-green), #059669)'
                                    : msg.role === 'error'
                                        ? 'var(--accent-red-dim)'
                                        : 'var(--bg-secondary)',
                                color: msg.role === 'user' ? 'white' : msg.role === 'error' ? 'var(--accent-red)' : 'var(--text-primary)',
                                fontSize: '0.9rem',
                                lineHeight: 1.6,
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                                border: msg.role === 'user' ? 'none' : '1px solid var(--border)',
                            }}>
                                {msg.role === 'assistant' && (
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                                        🤖 EV Assistant
                                    </span>
                                )}
                                {msg.content}
                            </div>
                        </div>
                    ))}

                    {loading && (
                        <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                            <div style={{
                                padding: '0.75rem 1rem', borderRadius: '16px 16px 16px 4px',
                                background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                                display: 'flex', alignItems: 'center', gap: '0.5rem'
                            }}>
                                <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Thinking...</span>
                            </div>
                        </div>
                    )}
                    <div ref={chatEndRef} />
                </div>

                {/* Suggestions */}
                {messages.length <= 2 && suggestions.length > 0 && (
                    <div style={{
                        padding: '0.75rem 1.25rem', borderTop: '1px solid var(--border)',
                        display: 'flex', flexWrap: 'wrap', gap: '0.5rem'
                    }}>
                        {suggestions.slice(0, 4).map((s, i) => (
                            <button key={i} className="btn btn-sm btn-secondary"
                                onClick={() => sendMessage(s)}
                                style={{ fontSize: '0.75rem' }}>
                                {s}
                            </button>
                        ))}
                    </div>
                )}

                {/* Input */}
                <div style={{
                    padding: '1rem 1.25rem', borderTop: '1px solid var(--border)',
                    display: 'flex', gap: '0.75rem', background: 'var(--bg-secondary)'
                }}>
                    <input
                        ref={inputRef}
                        className="form-control"
                        placeholder="Ask about EVs, charging, batteries..."
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={loading}
                        style={{ flex: 1, fontSize: '0.95rem', padding: '0.75rem 1rem' }}
                    />
                    <button className="btn btn-primary" onClick={() => sendMessage()} disabled={loading || !input.trim()}
                        style={{ padding: '0.75rem 1.5rem' }}>
                        {loading ? '⏳' : '🚀'} Send
                    </button>
                </div>
            </div>
        </div>
    )
}
