"""
EV Chatbot API Routes — ChatGroq LLM with EV FAQ context
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import os

chatbot_bp = Blueprint('chatbot', __name__)

# Conversation history per session (in-memory)
_conversations = {}

# System prompt with EV domain knowledge
SYSTEM_PROMPT = """You are an expert EV (Electric Vehicle) charging station assistant for India. 
You help users with:
- EV charging station information (types, connectors, power levels)
- EV battery maintenance and care tips
- Charging costs and time estimates
- EV buying guidance for Indian market
- Locating and comparing EV charging stations
- Understanding different charger types (AC Level 1/2, DC Fast, CCS2, CHAdeMO, Type 2)
- Government subsidies and policies for EVs in India (FAME II, state policies)
- Range anxiety solutions and trip planning

Be concise, helpful, and friendly. Use bullet points when listing information.
If asked about specific station locations, recommend using our Location Planner or Route Finder modules.
Always relate answers to the Indian EV ecosystem when relevant.
"""


def load_faq_context():
    """Load FAQ dataset as additional context."""
    data_dir = current_app.config['DATA_DIR']
    path = os.path.join(data_dir, 'ev_maintenance_faq.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        faqs = []
        for _, row in df.head(50).iterrows():
            q = row.get('question', row.get('Question', ''))
            a = row.get('answer', row.get('Answer', ''))
            if q and a:
                faqs.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(faqs)
    return ""


@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """Send a message to the chatbot."""
    data = request.json
    message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    api_key = data.get('api_key', '')

    if not message:
        return jsonify({'error': 'Message is required'}), 400
    if not api_key:
        return jsonify({'error': 'API key is required. Get one from https://console.groq.com'}), 400

    # Initialize conversation history
    if session_id not in _conversations:
        faq_context = load_faq_context()
        system_msg = SYSTEM_PROMPT
        if faq_context:
            system_msg += f"\n\nHere is an EV FAQ knowledge base you can reference:\n{faq_context}"
        _conversations[session_id] = [
            {"role": "system", "content": system_msg}
        ]

    # Add user message
    _conversations[session_id].append({"role": "user", "content": message})

    # Keep conversation manageable (last 20 messages + system)
    if len(_conversations[session_id]) > 21:
        _conversations[session_id] = [_conversations[session_id][0]] + _conversations[session_id][-20:]

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=_conversations[session_id],
            temperature=0.7,
            max_tokens=1024,
        )

        reply = completion.choices[0].message.content

        # Add assistant reply to history
        _conversations[session_id].append({"role": "assistant", "content": reply})

        return jsonify({
            'reply': reply,
            'model': 'llama-3.3-70b-versatile',
            'session_id': session_id,
        })

    except Exception as e:
        error_msg = str(e)
        if 'invalid_api_key' in error_msg.lower() or 'authentication' in error_msg.lower():
            return jsonify({'error': 'Invalid API key. Please check your Groq API key.'}), 401
        return jsonify({'error': f'ChatGroq error: {error_msg}'}), 500


@chatbot_bp.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history."""
    session_id = request.json.get('session_id', 'default')
    if session_id in _conversations:
        del _conversations[session_id]
    return jsonify({'cleared': True, 'session_id': session_id})


@chatbot_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Quick question suggestions."""
    return jsonify([
        "What types of EV chargers are available in India?",
        "How do I maintain my EV battery?",
        "What is the cost of charging an EV?",
        "Explain CCS2 vs CHAdeMO connectors",
        "What government subsidies are available for EVs?",
        "How long does it take to fully charge an EV?",
        "What is the range of popular EVs in India?",
        "Tips for reducing range anxiety",
    ])
