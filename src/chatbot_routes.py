import os
import re
import sys
import json
import uuid
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import Counter, defaultdict
from flask import Blueprint, request, jsonify, session, render_template
from jinja2 import TemplateNotFound  # Correct import
import faiss
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import pickle

# === Blueprint ===
chatbot_bp = Blueprint("chatbot", __name__, template_folder="../templates")  # Relative to src/

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Config Paths ===
# Paths relative to project root (D:\ai project\global-uk-tariffv1\)
CSV_PATH = "global-uk-tariff - Copy (2).xlsx"
FAISS_INDEX_PATH = "hs_code_index.faiss"
DF_PICKLE_PATH = "hs_code_data.pkl"
CACHE_FILE = "cache.pkl"
LOG_PATH = "chat_log.jsonl"

# === Domain Knowledge ===
STOPWORDS = set("""
a an and the of for in on with without to from by as or nor not other otherwise
including include etc etc., all any each every either neither both at is are was were be being been
this that these those such same different type types purpose purposes used use using
""".split())

MATERIALS = [
    "steel", "stainless steel", "stainless", "iron", "cast iron", "copper", "aluminium", "aluminum",
    "zinc", "nickel", "titanium", "magnesium", "lead", "tin", "brass", "bronze", "plastic", "polymer",
    "rubber", "wood", "paper", "cardboard", "ceramic", "glass", "textile", "leather", "fabric",
    "stone", "cement", "concrete", "gold", "silver", "platinum", "palladium", "composite", "alloy"
]

USES = [
    "automotive", "motor vehicle", "vehicle", "car", "truck", "railway", "aircraft", "aerospace",
    "marine", "ship", "boat", "construction", "building", "infrastructure", "furniture",
    "electrical", "electronics", "telecom", "medical", "surgical", "pharmaceutical", "agriculture",
    "mining", "oil", "gas", "food", "beverage", "packaging", "machine tools", "hand tools",
    "household", "industrial", "office", "school", "laboratory", "HVAC", "plumbing", "sanitary",
    "solar", "battery", "printing", "computing", "telecommunication", "audio", "video", "optical",
    "consumer", "commercial", "residential"
]

FEATURES = [
    "threaded", "non-threaded", "coated", "plated", "galvanised", "galvanized", "zinc plated",
    "cold-rolled", "hot-rolled", "forged", "cast", "welded", "seamless", "alloy", "non-alloy",
    "refined", "unrefined", "polished", "anodized", "painted", "lacquered", "insulated",
    "waterproof", "fireproof", "antistatic", "magnetic", "non-magnetic", "self-tapping", "hexagonal"
]

NON_DISCRIMINATIVE_WORDS = {
    'forms', 'solid', 'containing', 'principally', 'designed', 'excl', 'including',
    'other', 'parts', 'use', 'kind', 'type', 'types', 'purpose', 'purposes'
}

SYNONYMS = {
    'aluminum': 'aluminium', 'color': 'colour', 'center': 'centre', 'meter': 'metre',
    'program': 'programme', 'tire': 'tyre', 'analyzer': 'analyser', 'defense': 'defence',
    'license': 'licence', 'practice': 'practise', 'organization': 'organisation'
}

CATEGORY_TEMPLATES = {
    'food': [
        {"key": "food_type", "question": "What type of food product is it?", "options": ["solid", "liquid", "powder", "concentrate"]},
        {"key": "food_purpose", "question": "What is its primary purpose?", "options": ["human consumption", "animal feed", "industrial use"]},
        {"key": "food_packaging", "question": "How is it packaged?", "options": ["bulk", "retail", "hermetically sealed"]}
    ],
    'machinery': [
        {"key": "power_type", "question": "What powers it?", "options": ["electric", "hydraulic", "pneumatic", "manual", "engine"]},
        {"key": "machine_function", "question": "What does it do?", "options": ["cutting", "forming", "joining", "moving", "processing"]}
    ],
    'electronics': [
        {"key": "voltage", "question": "What voltage range?", "options": ["low voltage", "medium voltage", "high voltage"]},
        {"key": "application", "question": "Where is it used?", "options": ["consumer", "industrial", "medical", "telecom"]}
    ],
    'textiles': [
        {"key": "fiber_type", "question": "What fiber material?", "options": ["natural", "synthetic", "blended"]},
        {"key": "textile_form", "question": "What form is it in?", "options": ["yarn", "fabric", "finished product"]}
    ]
}

# === Cache Management ===
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
        logger.error(f"Error loading cache from {CACHE_FILE}: {e}")
        cache = {}
        save_cache()

def save_cache():
    global cache
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.error(f"Error saving cache to {CACHE_FILE}: {e}")

# === Utility Functions ===
def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [w for w in normalize_text(s).split() if w not in STOPWORDS and len(w) > 1]

def normalize_synonyms(text: str) -> str:
    words = text.split()
    return ' '.join(SYNONYMS.get(word, word) for word in words)

def extract_facets(desc: str) -> Dict[str, Any]:
    d = normalize_text(desc)
    d = normalize_synonyms(d)
    facets = {
        "materials": sorted({m for m in MATERIALS if m in d}),
        "uses": sorted({u for u in USES if u in d}),
        "features": sorted({f for f in FEATURES if f in d}),
        "threaded": bool(re.search(r"\bthread(ed|ing)?\b", d)),
        "coated": bool(re.search(r"\b(zinc|galvanis|coated|plated)\b", d)),
        "stainless": "stainless" in d,
        "dimensions_hint": bool(re.search(r"\b(mm|cm|m|inch|in|dia|diameter|length)\b", d)),
        "weight_hint": bool(re.search(r"\b(kg|g|lb|pound|ounce)\b", d)),
        "food_related": bool(re.search(r"\b(food|milk|cream|butter|cheese|meat|fruit|vegetable|grain)\b", d)),
        "mechanical_related": bool(re.search(r"\b(engine|motor|gear|bearing|valve|pump|compressor)\b", d)),
        "electrical_related": bool(re.search(r"\b(voltage|current|circuit|transformer|capacitor|resistor)\b", d)),
        "textile_related": bool(re.search(r"\b(fabric|textile|yarn|fiber|cloth|woven|knitted)\b", d))
    }
    return facets

def detect_category(facets: Dict[str, Any]) -> Optional[str]:
    if facets["food_related"]:
        return "food"
    elif facets["mechanical_related"]:
        return "machinery"
    elif facets["electrical_related"]:
        return "electronics"
    elif facets["textile_related"]:
        return "textiles"
    return None

def fuzzy_match_answer(answer: str, options: List[str], threshold: int = 80) -> Optional[str]:
    answer = normalize_synonyms(answer.lower())
    for opt in options:
        if fuzz.ratio(answer, normalize_synonyms(opt.lower())) >= threshold:
            return opt
    return None

# === HSCodeAssistant Class ===
class HSCodeAssistant:
    def __init__(self, df_path: str, index_path: str):
        try:
            if not os.path.exists(df_path):
                raise FileNotFoundError(f"Data file not found: {df_path}")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

            self.df = pd.read_pickle(df_path)
            self.index = faiss.read_index(index_path)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Ensure 'commodity' and 'description' are strings, handle NaN
            if 'commodity' not in self.df.columns or 'description' not in self.df.columns:
                raise ValueError("DataFrame missing required columns: 'commodity' or 'description'")
            
            self.df['commodity'] = self.df['commodity'].astype(str).replace('nan', '')
            self.df['description'] = self.df['description'].astype(str).replace('nan', '')

            # Validate commodity values
            invalid_commodities = self.df['commodity'].str.len() < 2
            if invalid_commodities.any():
                logger.warning(f"Found {invalid_commodities.sum()} invalid commodity codes (empty or too short)")
                self.df.loc[invalid_commodities, 'commodity'] = ''

            # Extract hierarchy
            self.df["chapter"] = self.df["commodity"].str.extract(r'(\d{2})')[0].fillna('')
            self.df["heading"] = self.df["commodity"].str.extract(r'(\d{4})')[0].fillna('')
            self.df["subheading"] = self.df["commodity"].str.extract(r'(\d{6})')[0].fillna('')

            # Build keyword index
            self.keyword_index = defaultdict(list)
            for idx, desc in enumerate(self.df['description']):
                for token in tokenize(normalize_synonyms(desc)):
                    self.keyword_index[token].append(idx)
            
            # Build hierarchy maps
            self.chapters_map = defaultdict(list)
            self.headings_map = defaultdict(list)
            self.subheadings_map = defaultdict(list)
            for idx, row in self.df.iterrows():
                if row['chapter']:
                    self.chapters_map[row['chapter']].append(idx)
                if row['heading']:
                    self.headings_map[row['heading']].append(idx)
                if row['subheading']:
                    self.subheadings_map[row['subheading']].append(idx)
            
            # Build category map
            self.category_map = defaultdict(list)
            for idx, desc in enumerate(self.df['description']):
                facets = extract_facets(desc)
                category = detect_category(facets)
                if category:
                    self.category_map[category].append(idx)
            
            self.feedback_data = []
            logger.info(f"Loaded model with {len(self.df)} HS codes and FAISS index")
        except FileNotFoundError as e:
            logger.error(f"Initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def hybrid_retrieve(self, query: str, top_k: int = 20) -> Tuple[List[int], np.ndarray]:
        try:
            query_embedding = np.array([self.embedding_model.encode(query)], dtype='float32')
            _, I = self.index.search(query_embedding, top_k)
            semantic_similarities = np.zeros(len(self.df))
            for i, idx in enumerate(I[0]):
                semantic_similarities[idx] = 1.0 - (i / top_k)  # Approximate similarity based on rank

            query_tokens = set(tokenize(query))
            keyword_boost = np.zeros(len(self.df))
            for token in query_tokens:
                if token in self.keyword_index:
                    for idx in self.keyword_index[token]:
                        keyword_boost[idx] += 0.3

            hs_pattern_boost = np.zeros(len(self.df))
            hs_digits = re.sub(r'\D', '', query)
            if len(hs_digits) >= 2:
                chapter = hs_digits[:2]
                if chapter in self.chapters_map:
                    for idx in self.chapters_map[chapter]:
                        hs_pattern_boost[idx] += 0.2

            category_boost = np.zeros(len(self.df))
            query_facets = extract_facets(query)
            detected_category = detect_category(query_facets)
            if detected_category and detected_category in self.category_map:
                for idx in self.category_map[detected_category]:
                    category_boost[idx] += 0.15

            combined_scores = (
                semantic_similarities * 0.55 +
                keyword_boost * 0.3 +
                hs_pattern_boost * 0.1 +
                category_boost * 0.05
            )

            idx = np.argsort(-combined_scores)[:top_k]
            return idx.tolist(), combined_scores[idx]
        except Exception as e:
            logger.error(f"Error in hybrid_retrieve: {e}")
            return [], np.array([])

    def propose_intelligent_questions(self, cand_idx: List[int], max_q: int = 3) -> List[Dict[str, Any]]:
        if len(cand_idx) <= 1:
            return []

        top_candidates = cand_idx[:min(8, len(cand_idx))]
        candidate_info = [
            {
                'hs_code': self.df.iloc[i]['commodity'],
                'description': self.df.iloc[i]['description'],
                'tokens': set(tokenize(self.df.iloc[i]['description']))
            }
            for i in top_candidates
        ]

        questions = []
        categories = {detect_category(extract_facets(cand['description'])) for cand in candidate_info}
        if len(categories) == 1 and None not in categories:
            category = next(iter(categories))
            if category in CATEGORY_TEMPLATES:
                questions.extend(CATEGORY_TEMPLATES[category][:2])

        materials = set()
        uses = set()
        features = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'])
            materials.update(facets['materials'])
            uses.update(facets['uses'])
            features.update(facets['features'])

        if len(materials) > 1 and len(questions) < max_q:
            questions.append({
                "key": "material",
                "question": "What material is it made of?",
                "options": sorted(materials),
                "type": "material"
            })

        if len(uses) > 1 and len(questions) < max_q:
            questions.append({
                "key": "use",
                "question": "What is its primary use or application?",
                "options": sorted(uses),
                "type": "use"
            })

        if len(features) > 1 and len(questions) < max_q:
            questions.append({
                "key": "feature",
                "question": "Does it have any of these features?",
                "options": sorted(features),
                "type": "feature"
            })

        if len(questions) < max_q:
            all_tokens = set()
            candidate_tokens = []
            for cand in candidate_info:
                tokens = cand['tokens'] - STOPWORDS - NON_DISCRIMINATIVE_WORDS
                tokens = {t for t in tokens if len(t) > 4 and not t.isdigit() and not t.endswith('ly') and not t.endswith('ing')}
                candidate_tokens.append(tokens)
                all_tokens.update(tokens)

            discriminating_terms = []
            for i, tokens_i in enumerate(candidate_tokens):
                for j, tokens_j in enumerate(candidate_tokens):
                    if i != j:
                        unique_tokens = tokens_i - tokens_j
                        for token in unique_tokens:
                            discriminating_terms.append((token, candidate_info[i]['hs_code']))

            term_counter = Counter([term for term, _ in discriminating_terms])
            for term, _ in term_counter.most_common(max_q - len(questions)):
                if term not in NON_DISCRIMINATIVE_WORDS and len(term) > 4 and not term.isdigit():
                    questions.append({
                        "key": f"term:{term}",
                        "question": f"Is it '{term}'?",
                        "options": ["yes", "no"],
                        "type": "discriminator"
                    })

        return questions[:max_q]

    def refine_with_answers(self, cand_idx: List[int], answers: Dict[str, str]) -> List[int]:
        kept = []
        contradictions = []

        for i in cand_idx:
            desc = normalize_text(self.df.iloc[i]["description"])
            desc = normalize_synonyms(desc)
            keep = True

            for key, value in answers.items():
                if not value:
                    continue

                normalized_value = normalize_synonyms(value.lower())

                if key == "material":
                    if normalized_value not in desc:
                        keep = False
                        contradictions.append(f"Material '{value}' not found in description")
                        break
                elif key == "use":
                    if normalized_value not in desc:
                        keep = False
                        contradictions.append(f"Use '{value}' not found in description")
                        break
                elif key == "feature":
                    if normalized_value not in desc:
                        keep = False
                        contradictions.append(f"Feature '{value}' not found in description")
                        break
                elif key.startswith("term:"):
                    term = key.split(":", 1)[1]
                    has_term = term in desc
                    if (value.lower() == "yes" and not has_term) or (value.lower() == "no" and has_term):
                        keep = False
                        contradictions.append(f"Term '{term}' mismatch with answer '{value}'")
                        break
                elif key in ["food_type", "food_purpose", "food_packaging", "power_type", "machine_function",
                             "voltage", "application", "fiber_type", "textile_form"]:
                    if normalized_value not in desc:
                        keep = False
                        contradictions.append(f"Category-specific '{key}' value '{value}' not found")
                        break

            if keep:
                kept.append(i)

        if not kept and contradictions:
            logger.warning(f"Refinement eliminated all candidates. Contradictions: {contradictions}")
            return cand_idx  # Fallback to original candidates if contradictions

        return kept

    def explain_confidence(self, candidate_idx: int, query: str) -> Dict[str, Any]:
        try:
            candidate_desc = self.df.iloc[candidate_idx]["description"]
            candidate_hs = self.df.iloc[candidate_idx]["commodity"]
            query_embedding = np.array([self.embedding_model.encode(query)], dtype='float32')
            candidate_embedding = self.df.iloc[candidate_idx]['embedding']
            semantic_sim = float(np.dot(query_embedding, candidate_embedding) /
                               (np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)))
            query_tokens = set(tokenize(normalize_synonyms(query)))
            desc_tokens = set(tokenize(normalize_synonyms(candidate_desc)))
            keyword_matches = query_tokens.intersection(desc_tokens)
            chapter = self.df.iloc[candidate_idx]["chapter"]
            heading = self.df.iloc[candidate_idx]["heading"]
            facets = extract_facets(candidate_desc)
            category = detect_category(facets)

            return {
                "confidence_score": max(0.0, min(1.0, semantic_sim)),
                "semantic_similarity": float(semantic_sim),
                "keyword_matches": list(keyword_matches),
                "match_count": len(keyword_matches),
                "hierarchy": {"chapter": chapter, "heading": heading, "commodity": candidate_hs},
                "category": category,
                "explanation": (f"Matched {len(keyword_matches)} keywords with {semantic_sim:.1%} semantic similarity"
                                + (f" in {category} category" if category else ""))
            }
        except Exception as e:
            logger.error(f"Error in explain_confidence: {e}")
            return {"confidence_score": 0.0, "explanation": "Error calculating confidence"}

    def add_feedback(self, query: str, correct_hs: str, user_answers: Dict[str, str]):
        try:
            self.feedback_data.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "correct_hs": correct_hs,
                "user_answers": user_answers
            })
            if len(self.feedback_data) % 100 == 0:
                logger.info(f"Accumulated {len(self.feedback_data)} feedback items")
                # Placeholder for retraining logic
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")

# === Routes ===
@chatbot_bp.route('/', methods=['GET'])
def chatbot_ui():
    """Serve the React frontend template."""
    try:
        return render_template('index.html')
    except TemplateNotFound:
        logger.error("Template 'index.html' not found in templates/ directory")
        return jsonify({"error": "Frontend template not found"}), 404

@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    global cache
    try:
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"error": "Missing or invalid 'user_input' in request"}), 400

        user_input = data['user_input'].strip()
        if not user_input:
            return jsonify({"error": "Empty input received"}), 400

        session_id = request.headers.get('X-Session-ID', session.get('session_id', str(uuid.uuid4())))
        session['session_id'] = session_id

        if session_id not in cache:
            cache[session_id] = {"conversation": [], "step": None, "answers": {}, "candidates": [], "scores": []}
        session_data = cache[session_id]

        trigger_keywords = [
            'import', 'export', 'buying', 'selling', 'trading', 'shipping', 'sourcing', 'hs code', 'commodity',
            'tariff', 'classification', 'duty', 'vat', 'origin', 'rule of origin'
        ] + MATERIALS + USES + FEATURES

        user_lower = user_input.lower()
        ongoing_wizard = session_data.get("step") is not None
        trigger_match = any(keyword in user_lower for keyword in trigger_keywords)

        if ongoing_wizard and any(k in user_lower for k in ['import', 'export', 'buying', 'selling', 'trading', 'sourcing']):
            logger.info(f"Resetting session {session_id} due to new product intent")
            session_data = {"conversation": [], "step": None, "answers": {}, "candidates": [], "scores": [], "round": 0}
            cache[session_id] = session_data

        if trigger_match or ongoing_wizard:
            response = interactive_hscode_handler(session_id, user_input, cache)
            session_data["conversation"].append({"role": "assistant", "content": response})
            cache[session_id] = session_data
            save_cache()
            return jsonify(response)

        # Fallback to general response
        response = {"response": "Please describe a product for HS code classification.", "status": "general"}
        session_data["conversation"].append({"role": "user", "content": user_input})
        session_data["conversation"].append({"role": "assistant", "content": response["response"]})
        cache[session_id] = session_data
        save_cache()
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /ask route: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def interactive_hscode_handler(session_id: str, user_input: str, cache: Dict) -> Dict[str, Any]:
    global assistant
    session_data = cache[session_id]
    step = session_data.get("step")

    try:
        if not step:
            product = normalize_text(user_input)
            cand_idx, scores = assistant.hybrid_retrieve(product, top_k=20)
            if not cand_idx:
                return {
                    "response": "No matches found. Try being more specific or using different terms.",
                    "status": "no_matches"
                }

            session_data.update({
                "step": "initial_retrieval",
                "product": product,
                "candidates": cand_idx,
                "scores": scores.tolist(),
                "answers": {},
                "round": 0
            })
            cache[session_id] = session_data
            save_cache()

            return {
                "response": f"Found {len(cand_idx)} potential matches",
                "candidates": format_candidates(assistant.df, cand_idx, scores, limit=5),
                "questions": assistant.propose_intelligent_questions(cand_idx),
                "status": "candidates_found"
            }

        elif step in ["initial_retrieval", "refining"]:
            answers = session_data.get("answers", {})
            round_num = session_data.get("round", 0)
            cand_idx = session_data.get("candidates", [])
            product = session_data.get("product", "")

            # Validate and normalize user answer
            questions = assistant.propose_intelligent_questions(cand_idx)
            if not questions:
                return finalize_classification(assistant, session_id, product, cand_idx, answers, cache)

            current_question = questions[0] if questions else None
            if current_question:
                normalized_answer = fuzzy_match_answer(user_input, current_question["options"])
                if not normalized_answer:
                    return {
                        "response": f"Invalid answer. Please choose from: {', '.join(current_question['options'])} or skip.",
                        "questions": questions,
                        "status": "invalid_answer"
                    }
                answers[current_question["key"]] = normalized_answer

            refined_idx = assistant.refine_with_answers(cand_idx, {current_question["key"]: normalized_answer} if current_question else {})
            if not refined_idx:
                logger.warning(f"Refinement failed for session {session_id}. Keeping original candidates.")
                return {
                    "response": "Answers eliminated all candidates. Keeping previous matches.",
                    "candidates": format_candidates(assistant.df, cand_idx, session_data["scores"], limit=5),
                    "questions": questions[1:] if len(questions) > 1 else assistant.propose_intelligent_questions(cand_idx),
                    "status": "refinement_failed"
                }

            enriched_query = product + " " + " ".join(answers.values())
            cand_idx, scores = assistant.hybrid_retrieve(enriched_query, top_k=len(refined_idx))
            round_num += 1

            session_data.update({
                "step": "refining" if len(cand_idx) > 1 and round_num < 2 else "final",
                "candidates": cand_idx,
                "scores": scores.tolist(),
                "answers": answers,
                "round": round_num
            })
            cache[session_id] = session_data
            save_cache()

            if len(cand_idx) > 1 and round_num < 2:
                return {
                    "response": f"Refined to {len(cand_idx)} candidates",
                    "candidates": format_candidates(assistant.df, cand_idx, scores, limit=5),
                    "questions": assistant.propose_intelligent_questions(cand_idx),
                    "status": "refining"
                }

            return finalize_classification(assistant, session_id, product, cand_idx, answers, cache)

        elif step == "final":
            product = session_data.get("product", "")
            cand_idx = session_data.get("candidates", [])
            answers = session_data.get("answers", {})
            feedback = user_input.lower()
            if feedback in ["yes", "no"]:
                correct_hs = assistant.df.iloc[cand_idx[0]]["commodity"] if feedback == "yes" else ""
                assistant.add_feedback(product, correct_hs, answers)
                session_data = {"conversation": session_data["conversation"], "step": None, "answers": {}, "candidates": [], "scores": [], "round": 0}
                cache[session_id] = session_data
                save_cache()
                return {
                    "response": "Thank you for your feedback! Describe another item to continue.",
                    "status": "feedback_received"
                }

            return {
                "response": "Please provide feedback ('yes', 'no', or 'skip') or describe another item.",
                "status": "awaiting_feedback"
            }

    except Exception as e:
        logger.error(f"Error in interactive_hscode_handler: {e}")
        return {"response": f"Error processing request: {str(e)}", "status": "error"}

def format_candidates(df: pd.DataFrame, idxs: List[int], scores: np.ndarray, limit: int = 5) -> List[Dict]:
    candidates = []
    for n, i in enumerate(idxs[:limit], 1):
        row = df.iloc[i]
        facets = extract_facets(row['description'])
        category = detect_category(facets)
        candidates.append({
            "rank": n,
            "hs_code": row['commodity'],
            "description": row['description'],
            "confidence": float(scores[n-1]) if n-1 < len(scores) else 0.0,
            "category": category.title() if category else None,
            "details": {
                "cet_duty_rate": str(row.get('cet_duty_rate', 'N/A')),
                "ukgt_duty_rate": str(row.get('ukgt_duty_rate', 'N/A')),
                "vat_rate": str(row.get('VAT Rate', 'N/A'))
            }
        })
    return candidates

def finalize_classification(assistant: HSCodeAssistant, session_id: str, product: str, cand_idx: List[int], answers: Dict, cache: Dict) -> Dict[str, Any]:
    if not cand_idx:
        session_data = cache[session_id]
        session_data.update({"step": None, "answers": {}, "candidates": [], "scores": [], "round": 0})
        cache[session_id] = session_data
        save_cache()
        return {
            "response": "No matches found after refinement.",
            "status": "no_matches"
        }

    final_idx = cand_idx[0]
    confidence = assistant.explain_confidence(final_idx, product + " " + " ".join(answers.values()))
    row = assistant.df.iloc[final_idx]

    response = {
        "response": f"HS Code: {row['commodity']}\nDescription: {row['description']}\nConfidence: { conference['confidence_score']:.1%}\nExplanation: {confidence['explanation']}",
        "result": {
            "hs_code": row['commodity'],
            "description": row['description'],
            "confidence": float(confidence['confidence_score']),
            "category": confidence['category'].title() if confidence['category'] else None,
            "details": {
                "cet_duty_rate": str(row.get('cet_duty_rate', 'N/A')),
                "ukgt_duty_rate": str(row.get('ukgt_duty_rate', 'N/A')),
                "vat_rate": str(row.get('VAT Rate', 'N/A')),
                "rule_of_origin_eu_uk": str(row.get('Product-specific rule of origin', 'N/A')),
                "rule_of_origin_japan": str(row.get('Product-specific rule of origin japan', 'N/A'))
            }
        },
        "status": "final"
    }

    session_data = cache[session_id]
    session_data.update({"step": "final", "candidates": [final_idx], "scores": [confidence['confidence_score']]})
    cache[session_id] = session_data
    save_cache()

    return response

# === Initialization ===
try:
    # Validate file paths
    for path in [DF_PICKLE_PATH, FAISS_INDEX_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    assistant = HSCodeAssistant(DF_PICKLE_PATH, FAISS_INDEX_PATH)
    load_cache()
    logger.info("Chatbot backend initialized successfully")
except FileNotFoundError as e:
    logger.error(f"Critical initialization failure: {e}. Ensure {DF_PICKLE_PATH} and {FAISS_INDEX_PATH} exist.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    sys.exit(1)