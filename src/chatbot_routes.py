import os
import json
import logging
import pickle
from datetime import datetime
from flask import Flask, Blueprint, request, jsonify, session, render_template
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import openai
import re
from hs_chat import HSCodeAssistant  # Reference hs_chat.py v4.4.2
import numpy as np
# Disable Elastic APM to prevent ConnectTimeoutError
try:
    import elasticapm
    elasticapm.instrument()  # No-op if not configured
except ImportError:
    pass

# Flask App and Blueprint
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
chatbot_bp = Blueprint("chatbot", __name__, template_folder="templates", static_folder="static")

# Config Paths
MODEL_PATH = "tariff_model.pkl"
CACHE_FILE = "cache.pkl"
LOG_PATH = "chat.log"
FEEDBACK_PATH = "feedback.jsonl"
ENCRYPTED_KEY_FILE = "encrypted_api_key_parts.txt"
KEY_FILE = "encryption_key.key"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# Load Environment
load_dotenv()

# Initialize Logger
logger = logging.getLogger('klynnai')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_PATH)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# OpenAI Key Decryption
def load_encryption_key():
    try:
        with open(KEY_FILE, "rb") as key_file:
            return key_file.read()
    except Exception as e:
        logger.error(f"Failed to load encryption key: {e}")
        raise

def decrypt_api_key_parts(encrypted_file_path, encryption_key):
    cipher = Fernet(encryption_key)
    try:
        with open(encrypted_file_path, "rb") as encrypted_file:
            encrypted_data = encrypted_file.read().split(b'\n')
            return cipher.decrypt(encrypted_data[0]).decode() + cipher.decrypt(encrypted_data[1]).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        raise

try:
    encryption_key = load_encryption_key()
    openai.api_key = decrypt_api_key_parts(ENCRYPTED_KEY_FILE, encryption_key)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI: {e}")
    raise

# Load Model
try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    bundle['model_path'] = MODEL_PATH
    hs_assistant = HSCodeAssistant(bundle)
    logger.info(f"Loaded model with {len(bundle['df_records'])} HS codes")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# In-Memory Cache
cache = {}

def load_cache():
    global cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}
            save_cache()

def save_cache():
    global cache
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Utility Functions
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    return re.sub(r'[^\w\s.,!?]', '', text.strip())

def get_embedding(text: str) -> list:
    try:
        response = openai.Embedding.create(input=text, model=OPENAI_EMBEDDING_MODEL)
        return response['data'][0]['embedding']
    except Exception as e:
        logger.error(f"OpenAI embedding error: {e}")
        return None

def format_hscode_results(row, confidence: dict) -> dict:
    """Format HS code result for JSON response."""
    # Handle None or incomplete confidence
    confidence = confidence or {'confidence_score': 0, 'explanation': 'No explanation available', 'category': 'N/A'}
    category = confidence.get('category', 'N/A')
    if category is None:
        category = 'N/A'
    return {
        "hs_code": row['commodity'],
        "description": row['description'],
        "confidence": f"{confidence.get('confidence_score', 0):.1f}%",
        "explanation": confidence.get('explanation', 'No explanation available'),
        "category": category.title() if isinstance(category, str) else category,
        "details": {
            "cet_duty_rate": row.get('cet_duty_rate', 'N/A'),
            "ukgt_duty_rate": row.get('ukgt_duty_rate', 'N/A'),
            "vat_rate": row.get('VAT Rate', 'N/A')
        }
    }

# Chat Logic
def interactive_hscode_handler(session_id: str, user_input: str, cache: dict) -> dict:
    user_input = sanitize_input(user_input)
    session_data = cache.get(session_id, {"conversation": [], "step": None, "answers": {}, "cand_idx": [], "scores": []})
    
    if session_data.get("step") is None:
        # Initial query
        if len(user_input) < 3:
            return {"response": "Please provide a detailed description (at least 3 characters).", "type": "error"}
        
        logger.info(f"Processing query: '{user_input}'")
        
        try:
            cand_idx, scores = hs_assistant.hybrid_retrieve(user_input, top_k=15)
            if not cand_idx:
                return {
                    "response": "No matches found. Try including material, use, or specific features. Example: 'stainless steel bolt for automotive' or 'milk for human consumption'",
                    "type": "error"
                }
            
            candidates = [{"hs_code": hs_assistant.df.iloc[i]["commodity"], "description": hs_assistant.df.iloc[i]["description"], "confidence": f"{scores[n]:.1f}%"} for n, i in enumerate(cand_idx[:5])]
            questions = hs_assistant.propose_intelligent_questions(cand_idx, set(), {}, max_q=3)
            
            session_data.update({
                "step": "refinement",
                "product": user_input,
                "cand_idx": cand_idx,
                "scores": scores.tolist(),
                "answers": {},
                "answered_keys": set(),
                "round": 0
            })
            cache[session_id] = session_data
            save_cache()
            
            return {
                "response": f"Found {len(cand_idx)} potential matches",
                "type": "candidates",
                "candidates": candidates,
                "questions": questions
            }
        
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return {"response": f"Error occurred: {e}. Please try again.", "type": "error"}
    
    elif session_data.get("step") == "refinement":
        # Handle refinement answers
        round = session_data.get("round", 0)
        answers = session_data.get("answers", {})
        answered_keys = session_data.get("answered_keys", set())
        cand_idx = session_data.get("cand_idx", [])
        scores = np.array(session_data.get("scores", []))
        
        if round >= 2 or len(cand_idx) <= 1:
            # Finalize
            cand_idx, scores = hs_assistant.hybrid_retrieve(session_data["product"], answers, top_k=len(cand_idx))
            final_idx = cand_idx[0]
            confidence = hs_assistant.explain_confidence(final_idx, session_data["product"], answers)
            logger.info(f"Confidence output: {confidence}")
            row = hs_assistant.df.iloc[final_idx]
            
            session_data.update({"step": "feedback", "final_idx": final_idx, "confidence": confidence})
            cache[session_id] = session_data
            save_cache()
            
            return {
                "response": "Final HS Code Classification",
                "type": "final",
                "result": format_hscode_results(row, confidence),
                "feedback_prompt": "Was this classification correct? (yes/no/skip)"
            }
        
        # Process user answers
        try:
            if user_input.startswith("{"):
                user_answers = json.loads(user_input)
            else:
                user_answers = {q["key"]: sanitize_input(v.lower()) for q, v in zip(session_data.get("questions", []), user_input.split(",")) if v.strip()}
            
            for key, value in user_answers.items():
                if value in ['skip', 'none', 'unknown']:
                    continue
                # Validate against options directly
                for q in session_data.get("questions", []):
                    if q["key"] == key and q.get("options"):
                        if value not in q["options"]:
                            logger.info(f"Invalid answer for '{q['question']}': '{value}' corrected to '{q['options'][0]}'")
                            value = q['options'][0]
                        answers[key] = value
                        answered_keys.add(key)
            
            refined = hs_assistant.refine_with_answers(cand_idx, answers)
            if len(refined) < len(cand_idx):
                cand_idx = refined
                cand_idx, scores = hs_assistant.hybrid_retrieve(session_data["product"], answers, top_k=len(cand_idx))
            
            questions = hs_assistant.propose_intelligent_questions(cand_idx, answered_keys, answers, max_q=3)
            candidates = [{"hs_code": hs_assistant.df.iloc[i]["commodity"], "description": hs_assistant.df.iloc[i]["description"], "confidence": f"{scores[n]:.1f}%"} for n, i in enumerate(cand_idx[:3])]
            
            session_data.update({
                "cand_idx": cand_idx,
                "scores": scores.tolist(),
                "answers": answers,
                "answered_keys": answered_keys,
                "round": round + 1,
                "questions": questions
            })
            cache[session_id] = session_data
            save_cache()
            
            return {
                "response": f"Refined to {len(cand_idx)} candidates",
                "type": "candidates",
                "candidates": candidates,
                "questions": questions
            }
        
        except Exception as e:
            logger.error(f"Error during refinement: {e}")
            return {"response": f"Error processing answers: {e}. Please try again.", "type": "error"}
    
    elif session_data.get("step") == "feedback":
        feedback_input = sanitize_input(user_input.lower())
        if feedback_input not in ['yes', 'no', 'skip']:
            return {"response": "Please enter 'yes', 'no', or 'skip'.", "type": "error"}
        
        correct_hs_code = None
        if feedback_input == 'no':
            correct_input = request.json.get('correct_hs_code', '')
            if correct_input:
                correct_hs_code = sanitize_input(correct_input)
        
        hs_assistant.add_feedback(
            session_data["product"],
            hs_assistant.df.iloc[session_data["final_idx"]]["commodity"],
            session_data.get("answers", {}),
            feedback_input,
            correct_hs_code
        )
        
        logger.info(f"Feedback received: {feedback_input}, Correct HS: {correct_hs_code}")
        
        conversation = session_data.get("conversation", [])
        cache[session_id] = {"conversation": conversation}
        save_cache()
        
        return {
            "response": "Thank you for your feedback!" + (" Your input will help improve future classifications." if feedback_input == 'no' else ""),
            "type": "feedback"
        }

# /ask Route
@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "Empty input received."}), 400
    
    session_id = session.get('session_id', os.urandom(16).hex())
    session['session_id'] = session_id
    
    if session_id not in cache:
        cache[session_id] = {"conversation": []}
    
    session_data = cache[session_id]
    conversation = session_data.get("conversation", [])
    
    trigger_keywords = [
        'import', 'export', 'buying', 'selling', 'trading', 'shipping', 'sending',
        'dealing with', 'looking for', 'sourcing', 'purchasing', 'hs code', 'commodity',
        'tariff', 'classification', 'chapter', 'heading', 'schedule b', 'duty', 'vat',
        'excise', 'origin', 'rule of origin', 'product origin', 'non-originating',
        'originating', 'preferential', 'zero duty', 'third country', 'trade agreement',
        'fta', 'customs code', 'cn code', 'steel', 'plastic', 'aluminum', 'rubber',
        'glass', 'wood', 'textile', 'leather', 'machinery', 'electrical', 'construction',
        'parts', 'equipment', 'raw material', 'components', 'bolts', 'screws', 'nuts',
        'fasteners', 'milk', 'lawn mower', 'pharmaceutical'
    ]
    
    user_lower = user_input.lower()
    ongoing_wizard = session_data.get("step") is not None
    trigger_match = any(keyword in user_lower for keyword in trigger_keywords)
    
    if trigger_match or ongoing_wizard:
        response = interactive_hscode_handler(session_id, user_input, cache)
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})
        session_data["conversation"] = conversation
        cache[session_id] = session_data
        save_cache()
        return jsonify(response)
    
    conversation.append({"role": "user", "content": user_input})
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_CHAT_MODEL,
            messages=conversation,
            max_tokens=300,
            temperature=0.7
        )
        answer = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        answer = "AI is currently unavailable. Please try again shortly."
    
    conversation.append({"role": "assistant", "content": answer})
    session_data["conversation"] = conversation
    cache[session_id] = session_data
    save_cache()
    
    return jsonify({"response": answer, "type": "chat"})

# UI Route
@chatbot_bp.route('/', methods=['GET'])
def chatbot_ui():
    return render_template('index.html')

# Initialize Cache
load_cache()

# Register Blueprint (handled in main.py)
# app.register_blueprint(chatbot_bp)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)