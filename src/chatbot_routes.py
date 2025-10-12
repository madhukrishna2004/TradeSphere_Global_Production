import os
import sys
import uuid
import logging
import pickle
from flask import Blueprint, request, jsonify, session, render_template
from jinja2 import TemplateNotFound

# === Blueprint ===
chatbot_bp = Blueprint("chatbot", __name__, template_folder="../templates")

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chatbot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === Cache ===
CACHE_FILE = "cache.pkl"
cache = {}

def load_cache():
    global cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
            logger.info(f"Loaded cache from {CACHE_FILE}")
        else:
            cache = {}
            save_cache()
            logger.info(f"Created new cache file: {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# === Routes ===
@chatbot_bp.route('/', methods=['GET'])
def chatbot_ui():
    try:
        return render_template('index.html')
    except TemplateNotFound:
        logger.error("Template 'index.html' not found")
        return jsonify({"error": "Frontend template not found"}), 404

@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "Missing 'user_input'"}), 400

        user_input = data['user_input'].strip()
        if not user_input:
            return jsonify({"error": "Empty input"}), 400

        session_id = request.headers.get('X-Session-ID', session.get('session_id', str(uuid.uuid4())))
        session['session_id'] = session_id

        if session_id not in cache:
            cache[session_id] = {"conversation": []}

        session_data = cache[session_id]
        session_data["conversation"].append({"role": "user", "content": user_input})
        cache[session_id] = session_data
        save_cache()

        # Since we removed HSCodeAssistant, return a default response
        response = {
            "response": "HS code model not loaded. Describe your product for general guidance.",
            "status": "general"
        }
        session_data["conversation"].append({"role": "assistant", "content": response["response"]})
        save_cache()

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        return jsonify({"error": "Server error occurred"}), 500

# === Initialization ===
try:
    load_cache()
    logger.info("Chatbot backend initialized without model")
except Exception as e:
    logger.error(f"Critical initialization failure: {e}")
    sys.exit(1)
