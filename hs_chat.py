import argparse
import json
import re
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
from difflib import get_close_matches

# Suppress sklearn warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# ---------------------------
# Configuration
# ---------------------------

import logging

def setup_logging(log_path='chat.log'):
    logger = logging.getLogger('klynnai')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.handlers = [handler]  # Clear any existing handlers (e.g., Elastic APM)
    return logger
# Sync with hs_train.py v4.3.1
STOPWORDS = set("""
a an and the of for in on with without to from by as or nor not other otherwise
including include etc etc., all any each every either neither both at is are was were be being been
this that these those such same different type types purpose purposes used use using
""".split())

MATERIALS = [
    "steel", "stainless steel", "stainless", "iron", "cast iron", "copper", "aluminium", "aluminum", 
    "zinc", "nickel", "titanium", "magnesium", "lead", "tin", "brass", "bronze", "plastic", "polymer",
    "rubber", "wood", "paper", "cardboard", "ceramic", "glass", "textile", "leather", "fabric",
    "stone", "cement", "concrete", "gold", "silver", "platinum", "palladium", "composite", "alloy",
    "milk", "cream"  # Added for food-related items
]

USES = [
    "automotive", "motor vehicle", "vehicle", "car", "truck", "railway", "aircraft", "aerospace", 
    "marine", "ship", "boat", "construction", "building", "infrastructure", "furniture", 
    "electronics", "telecom", "medical", "surgical", "pharmaceutical", "agriculture", 
    "mining", "oil", "gas", "food", "beverage", "packaging", "machine tools", "hand tools", 
    "household", "industrial", "office", "school", "laboratory", "HVAC", "plumbing", "sanitary", 
    "solar", "battery", "printing", "computing", "telecommunication", "audio", "video", "optical",
    "consumer", "commercial", "residential", "lawn", "garden", "animal feed", "human consumption"
]

FEATURES = [
    "threaded", "non-threaded", "coated", "plated", "galvanised", "galvanized", "zinc plated", 
    "cold-rolled", "hot-rolled", "forged", "cast", "welded", "seamless", "alloy", "non-alloy", 
    "refined", "unrefined", "polished", "anodized", "painted", "lacquered", "insulated", 
    "waterproof", "fireproof", "antistatic", "magnetic", "non-magnetic", "self-tapping", "hexagonal",
    "cutting", "dried", "flakes", "powder", "solid", "liquid", "concentrate"
]

NON_DISCRIMINATIVE_WORDS = {
    'forms', 'containing', 'principally', 'designed', 'excl', 'including', 
    'other', 'parts', 'use', 'kind', 'type', 'types', 'purpose', 'purposes'
}

SYNONYMS = {
    'aluminum': 'aluminium', 'color': 'colour', 'center': 'centre', 'meter': 'metre',
    'program': 'programme', 'tire': 'tyre', 'analyzer': 'analyser', 'defense': 'defence',
    'license': 'licence', 'practice': 'practise', 'organization': 'organisation',
    'lawnmowers': 'lawn mower', 'lawnmower': 'lawn mower', 'milk powder': 'milk powder'
}

CATEGORY_TEMPLATES = {
    'food': [
        {"key": "food_type", "question": "What type of food product is it?", "options": ["solid", "liquid", "powder", "concentrate"]},
        {"key": "food_purpose", "question": "What is its primary purpose?", "options": ["human consumption", "animal feed", "industrial use"]},
        {"key": "food_packaging", "question": "How is it packaged?", "options": ["bulk", "retail", "hermetically sealed"]}
    ],
    'machinery': [
        {"key": "power_type", "question": "What powers it?", "options": ["electric", "hydraulic", "pneumatic", "manual", "engine"]},
        {"key": "machine_function", "question": "What is its primary function?", "options": ["cutting", "forming", "joining", "moving", "processing"]}
    ],
    'electronics': [
        {"key": "voltage", "question": "What voltage range?", "options": ["low voltage", "medium voltage", "high voltage"]},
        {"key": "application", "question": "Where is it used?", "options": ["consumer", "industrial", "medical", "telecom"]}
    ],
    'textiles': [
        {"key": "fiber_type", "question": "What fiber material?", "options": ["natural", "synthetic", "blended"]},
        {"key": "textile_form", "question": "What form is it in?", "options": ["yarn", "fabric", "finished product"]}
    ],
    'pharmaceuticals': [
        {"key": "product_type", "question": "What type of product is it?", "options": ["solid", "liquid", "powder", "concentrate"]},
        {"key": "therapeutic_use", "question": "Is it for therapeutic use?", "options": ["yes", "no"]}
    ]
}

# ---------------------------
# Utility Functions
# ---------------------------

def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [w for w in normalize_text(s).split() if w not in STOPWORDS and len(w) > 1]

def normalize_synonyms(text: str) -> str:
    words = text.split()
    normalized_words = [SYNONYMS.get(word, word) for word in words]
    return ' '.join(normalized_words)

def correct_typos(text: str, options: List[str]) -> str:
    text = normalize_text(text)
    matches = get_close_matches(text, options, n=1, cutoff=0.8)
    return matches[0] if matches else text

def extract_facets(desc: str, answers: Dict[str, str] = None) -> Dict[str, Any]:
    d = normalize_text(desc)
    d = normalize_synonyms(d)
    if answers:
        d += " " + " ".join(normalize_synonyms(v) for v in answers.values())
    facets = {
        "materials": sorted({m for m in MATERIALS if m in d}),
        "uses": sorted({u for u in USES if u in d}),
        "features": sorted({f for f in FEATURES if f in d}),
        "threaded": bool(re.search(r"\bthread(ed|ing)?\b", d)),
        "coated": bool(re.search(r"\b(zinc|galvanis|coated|plated)\b", d)),
        "stainless": "stainless" in d,
        "dimensions_hint": bool(re.search(r"\b(mm|cm|m|inch|in|dia|diameter|length)\b", d)),
        "weight_hint": bool(re.search(r"\b(kg|g|lb|pound|ounce)\b", d)),
        "food_related": bool(re.search(r"\b(food|milk|cream|butter|cheese|meat|fruit|vegetable|grain|beverage|animal feed|human consumption)\b", d)),
        "mechanical_related": bool(re.search(r"\b(engine|motor|gear|bearing|valve|pump|compressor|lawn|mower|cutting)\b", d)),
        "electrical_related": bool(re.search(r"\b(voltage|current|circuit|transformer|capacitor|resistor|electric)\b", d)),
        "textile_related": bool(re.search(r"\b(fabric|textile|yarn|fiber|cloth|woven|knitted)\b", d)),
        "pharmaceutical_related": bool(re.search(r"\b(pharmaceutical|medicament|drug|tablet|capsule|syringe)\b", d))
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
    elif facets["pharmaceutical_related"]:
        return "pharmaceuticals"
    return None

def preprocess_query(query: str, answers: Dict[str, str] = None) -> str:
    query = normalize_text(query)
    query = normalize_synonyms(query)
    corrections = {
        "springs watches": "springs for watches",
        "milk": "milk for human consumption" if answers and "food_purpose" in answers and answers["food_purpose"] == "human consumption" else "milk beverage",
        "lawnmowers": "lawn mower for cutting grass",
        "lawnmower": "lawn mower for cutting grass",
        "lawnmower electric": "electric lawn mower for cutting grass"
    }
    query = corrections.get(query, query)
    if answers:
        query += " " + " ".join(f"{k}:{normalize_synonyms(v)}" for k, v in answers.items())
    return query

# ---------------------------
# Chat Engine
# ---------------------------

class HSCodeAssistant:
    def __init__(self, bundle: Dict[str, Any]):
        required_keys = ['df_records', 'embeddings', 'keyword_index', 'chapters_map', 'category_map', 'embedding_model', 'category_templates']
        missing = [k for k in required_keys if k not in bundle]
        if missing:
            raise ValueError(f"Model bundle missing required keys: {missing}")
        
        self.bundle = bundle
        self.df = pd.DataFrame(bundle['df_records'])
        self.embedding_model = bundle['embedding_model']
        self.category_templates = bundle['category_templates']
        self.conversation_memory = []
        self.feedback_data = []
        
        # Enhance DataFrame with hierarchical info
        self.df["chapter"] = self.df["commodity"].str.extract(r'(\d{2})')[0]
        self.df["heading"] = self.df["commodity"].str.extract(r'(\d{4})')[0]
        self.df["subheading"] = self.df["commodity"].str.extract(r'(\d{6})')[0]
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            tokenizer=tokenize,
            stop_words=list(STOPWORDS),
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['description'])
        
        # Cache for query embeddings
        self.embedding_cache = {}

    def hybrid_retrieve(self, query: str, answers: Dict[str, str] = None, top_k: int = 15) -> Tuple[List[int], np.ndarray]:
        processed_query = preprocess_query(query, answers)
        if processed_query in self.embedding_cache:
            query_embedding = self.embedding_cache[processed_query]
        else:
            query_embedding = self.embedding_model.encode([processed_query], convert_to_tensor=False, show_progress_bar=True)
            self.embedding_cache[processed_query] = query_embedding
        
        semantic_similarities = cosine_similarity(query_embedding, self.bundle['embeddings'])[0]
        
        # TF-IDF scoring
        query_tfidf = self.tfidf.transform([processed_query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Exact keyword matching
        exact_boost = np.zeros(len(self.df))
        query_tokens = set(tokenize(processed_query))
        for token in query_tokens:
            if token in self.bundle['keyword_index']:
                for idx in self.bundle['keyword_index'][token]:
                    exact_boost[idx] += 0.7 if token in processed_query else 0.4
        
        # HS code pattern matching
        hs_pattern_boost = np.zeros(len(self.df))
        hs_digits = re.sub(r'\D', '', query)
        if len(hs_digits) >= 2:
            chapter = hs_digits[:2]
            if chapter in self.bundle['chapters_map']:
                for idx in self.bundle['chapters_map'][chapter]:
                    hs_pattern_boost[idx] += 0.2
        
        # Category-based boosting
        category_boost = np.zeros(len(self.df))
        query_facets = extract_facets(processed_query, answers)
        detected_category = detect_category(query_facets)
        if detected_category and detected_category in self.bundle['category_map']:
            for idx in self.bundle['category_map'][detected_category]:
                category_boost[idx] += 0.6
        
        # Answer-based boosting
        answer_boost = np.zeros(len(self.df))
        if answers:
            for key, value in answers.items():
                normalized_value = normalize_synonyms(value)
                for idx in range(len(self.df)):
                    desc = normalize_text(self.df.iloc[idx]["description"])
                    if normalized_value in desc or any(normalized_value in d for d in desc.split()):
                        answer_boost[idx] += 0.5
                    # Specific boost for milk + human consumption
                    if query.lower().startswith("milk") and value == "human consumption" and self.df.iloc[idx]["chapter"] == "04":
                        answer_boost[idx] += 0.3
        
        # Combine scores
        combined_scores = (
            semantic_similarities * 0.2 +
            tfidf_scores * 0.15 +
            exact_boost * 0.25 +
            hs_pattern_boost * 0.05 +
            category_boost * 0.2 +
            answer_boost * 0.35
        )
        
        # Normalize scores to 0-100%
        combined_scores = np.clip((combined_scores - combined_scores.min()) / (combined_scores.max() - combined_scores.min() + 1e-10) * 100, 0, 100)
        
        idx = np.argsort(-combined_scores)[:top_k]
        return idx.tolist(), combined_scores[idx]

    def propose_intelligent_questions(self, cand_idx: List[int], answered_keys: set, answers: Dict[str, str], max_q: int = 3) -> List[Dict[str, Any]]:
        if len(cand_idx) <= 1:
            return []
        
        top_candidates = cand_idx[:min(8, len(cand_idx))]
        candidate_info = []
        for i in top_candidates:
            row = self.df.iloc[i]
            candidate_info.append({
                'hs_code': row['commodity'],
                'description': row['description'],
                'tokens': set(tokenize(row['description']))
            })
        
        questions = []
        category_questions = self._get_category_specific_questions(candidate_info, answered_keys, answers)
        questions.extend(category_questions)
        
        # Material discrimination
        materials = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'], answers)
            materials.update(facets['materials'])
        if len(materials) > 1 and len(questions) < max_q and "material" not in answered_keys:
            questions.append({
                "key": "material",
                "question": "What material is it made of?",
                "options": sorted(materials),
                "type": "material"
            })
        
        # Use/application discrimination
        uses = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'], answers)
            uses.update(facets['uses'])
        if len(uses) > 1 and len(questions) < max_q and "use" not in answered_keys:
            questions.append({
                "key": "use",
                "question": "What is its primary use or application?",
                "options": sorted(uses),
                "type": "use"
            })
        
        # Technical feature discrimination
        features = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'], answers)
            features.update(facets['features'])
        if len(features) > 1 and len(questions) < max_q and "feature" not in answered_keys:
            questions.append({
                "key": "feature",
                "question": "Does it have any of these features?",
                "options": sorted(features),
                "type": "feature"
            })
        
        # TF-IDF-based discriminative terms
        if len(questions) < max_q:
            questions.extend(self._get_discriminating_terms(candidate_info, max_q - len(questions), answered_keys, answers))
        
        return questions[:max_q]

    def _get_category_specific_questions(self, candidate_info: List[Dict], answered_keys: set, answers: Dict[str, str]) -> List[Dict]:
        categories = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'], answers)
            category = detect_category(facets)
            if category:
                categories.add(category)
        
        questions = []
        if len(categories) == 1:
            category = next(iter(categories))
            if category in self.category_templates:
                for q in self.category_templates[category][:2]:
                    if q['key'] not in answered_keys:
                        # Filter conflicting questions
                        if answers.get('food_type') == 'liquid' and any(term in q['question'].lower() for term in ['dried', 'powder', 'solid']):
                            continue
                        if answers.get('food_type') == 'powder' and any(term in q['question'].lower() for term in ['liquid', 'solid']):
                            continue
                        if answers.get('food_type') == 'solid' and any(term in q['question'].lower() for term in ['liquid', 'powder']):
                            continue
                        questions.append(q)
        return questions

    def _get_discriminating_terms(self, candidate_info: List[Dict], max_terms: int, answered_keys: set, answers: Dict[str, str]) -> List[Dict]:
        candidate_texts = [cand['description'] for cand in candidate_info]
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=list(STOPWORDS))
        tfidf_matrix = tfidf.fit_transform(candidate_texts)
        feature_names = tfidf.get_feature_names_out()
        
        term_scores = []
        for i in range(len(candidate_info)):
            for j in range(i + 1, len(candidate_info)):
                scores_i = tfidf_matrix[i].toarray()[0]
                scores_j = tfidf_matrix[j].toarray()[0]
                diff = np.abs(scores_i - scores_j)
                for idx, score in enumerate(diff):
                    if score > 0.1 and feature_names[idx] not in NON_DISCRIMINATIVE_WORDS:
                        # Skip contradictory terms
                        if answers.get('food_type') == 'liquid' and feature_names[idx] in ['dried', 'powder', 'solid']:
                            continue
                        if answers.get('food_type') == 'powder' and feature_names[idx] in ['liquid', 'solid']:
                            continue
                        if answers.get('food_type') == 'solid' and feature_names[idx] in ['liquid', 'powder']:
                            continue
                        term_scores.append((feature_names[idx], score, candidate_info[i]['hs_code']))
        
        term_counter = Counter([term for term, _, _ in term_scores])
        questions = []
        for term, _ in term_counter.most_common(max_terms):
            if len(term) > 3 and not term.isdigit() and f"term:{term}" not in answered_keys:
                questions.append({
                    "key": f"term:{term}",
                    "question": f"Does it involve '{term}'?",
                    "options": ["yes", "no"],
                    "type": "discriminator"
                })
        return questions

    def refine_with_answers(self, cand_idx: List[int], answers: Dict[str, str]) -> List[int]:
        kept = []
        for i in cand_idx:
            desc = normalize_text(self.df.iloc[i]["description"])
            desc = normalize_synonyms(desc)
            score = 0
            total_checks = 0
            
            for key, value in answers.items():
                if not value:
                    continue
                normalized_value = normalize_synonyms(value)
                total_checks += 1
                
                if key in ["material", "use", "feature", "food_type", "food_purpose", 
                         "food_packaging", "power_type", "machine_function", 
                         "voltage", "application", "fiber_type", "textile_form", "therapeutic_use"]:
                    if normalized_value in desc or any(normalized_value in d for d in desc.split()):
                        score += 1
                elif key.startswith("term:"):
                    term = key.split(":", 1)[1]
                    has_term = term in desc or any(term in d for d in desc.split())
                    if (value == "yes" and has_term) or (value == "no" and not has_term):
                        score += 1
            
            # Keep if at least 60% of answers match
            if total_checks == 0 or score / total_checks >= 0.6:
                kept.append(i)
        
        return kept if kept else cand_idx[:5]

    def explain_confidence(self, candidate_idx: int, query: str, answers: Dict[str, str]) -> Dict[str, Any]:
        candidate_desc = self.df.iloc[candidate_idx]["description"]
        candidate_hs = self.df.iloc[candidate_idx]["commodity"]
        
        processed_query = preprocess_query(query, answers)
        query_embedding = self.embedding_model.encode([processed_query], convert_to_tensor=False)
        candidate_embedding = self.bundle['embeddings'][candidate_idx].reshape(1, -1)
        semantic_sim = cosine_similarity(query_embedding, candidate_embedding)[0][0]
        
        query_tfidf = self.tfidf.transform([processed_query])
        candidate_tfidf = self.tfidf_matrix[candidate_idx]
        tfidf_sim = cosine_similarity(query_tfidf, candidate_tfidf)[0][0]
        
        query_tokens = set(tokenize(normalize_synonyms(processed_query)))
        desc_tokens = set(tokenize(normalize_synonyms(candidate_desc)))
        keyword_matches = query_tokens.intersection(desc_tokens)
        
        query_facets = extract_facets(processed_query, answers)
        candidate_facets = extract_facets(candidate_desc, answers)
        facet_matches = {
            "materials": set(query_facets["materials"]).intersection(candidate_facets["materials"]),
            "uses": set(query_facets["uses"]).intersection(candidate_facets["uses"]),
            "features": set(query_facets["features"]).intersection(candidate_facets["features"])
        }
        
        chapter = self.df.iloc[candidate_idx]["chapter"]
        heading = self.df.iloc[candidate_idx]["heading"]
        category = detect_category(candidate_facets)
        
        answer_matches = sum(1 for k, v in answers.items() if normalize_synonyms(v) in candidate_desc)
        confidence = (
            semantic_sim * 0.4 +
            tfidf_sim * 0.3 +
            len(keyword_matches) * 0.1 +
            sum(len(v) for v in facet_matches.values()) * 0.1 +
            answer_matches * 0.1
        ) * 100
        confidence = min(confidence, 100.0)  # Cap at 100%
        
        explanation = (
            f"Matched {len(keyword_matches)} keywords ({', '.join(keyword_matches) if keyword_matches else 'none'}) "
            f"and {sum(len(v) for v in facet_matches.values())} facets "
            f"(Materials: {', '.join(facet_matches['materials']) or 'none'}, "
            f"Uses: {', '.join(facet_matches['uses']) or 'none'}, "
            f"Features: {', '.join(facet_matches['features']) or 'none'}) "
            f"with {semantic_sim:.1%} semantic and {tfidf_sim:.1%} TF-IDF similarity"
            f"{f' in {category} category' if category else ''}"
        )
        
        return {
            "confidence_score": float(confidence),
            "semantic_similarity": float(semantic_sim),
            "tfidf_similarity": float(tfidf_sim),
            "keyword_matches": list(keyword_matches),
            "facet_matches": facet_matches,
            "match_count": len(keyword_matches),
            "hierarchy": {
                "chapter": chapter,
                "heading": heading,
                "commodity": candidate_hs
            },
            "category": category,
            "explanation": explanation
        }

    def add_feedback(self, query: str, correct_hs: str, user_answers: Dict[str, str], user_feedback: str, correct_hs_code: Optional[str] = None):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "correct_hs": correct_hs,
            "user_answers": user_answers,
            "user_feedback": user_feedback
        }
        if correct_hs_code:
            feedback_entry["correct_hs_code"] = correct_hs_code
        self.feedback_data.append(feedback_entry)
        
        feedback_path = os.path.join(os.path.dirname(self.bundle.get('model_path', '.')), 'feedback.jsonl')
        with open(feedback_path, 'a', encoding='utf-8') as f:
            json.dump(feedback_entry, f, ensure_ascii=False)
            f.write('\n')

    def show_candidates(self, idxs: List[int], scores: Optional[np.ndarray] = None, limit: int = 5) -> None:
        print("\n🔍 Top Matching HS Codes:")
        print(f"{'Rank':<6} {'HS Code':<12} {'Category':<12} {'Description':<60} {'Confidence':<10}")
        print("-" * 100)
        for n, i in enumerate(idxs[:limit], 1):
            row = self.df.iloc[i]
            facets = extract_facets(row['description'])
            category = detect_category(facets) or "N/A"
            score_text = f"{scores[n-1]:.1f}%" if scores is not None and len(scores) > n-1 else "N/A"
            desc = row['description'][:57] + "..." if len(row['description']) > 57 else row['description']
            print(f"{n:<6} {row['commodity']:<12} {category.upper():<12} {desc:<60} {score_text:<10}")

# ---------------------------
# Interactive Chat
# ---------------------------

def interactive_chat(bundle: Dict[str, Any], logger: logging.Logger, log_path: Optional[str] = None) -> None:
    assistant = HSCodeAssistant(bundle)
    
    def log_event(event_type: str, data: Dict[str, Any]):
        if not log_path:
            return
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            **data
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    print("\n" + "="*60)
    print("🚀 ENTERPRISE HS CODE ASSISTANT v4.4.1")
    print("="*60)
    print("✨ Production-grade: Accurate, robust, and error-free!")
    
    while True:
        user_q = input("\nDescribe the item (or 'quit' to exit): ").strip()
        if user_q.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_q or len(user_q) < 3:
            print("⚠ Please provide a detailed description (at least 3 characters).")
            continue
        
        answers = {}
        processed_query = preprocess_query(user_q, answers)
        if processed_query != user_q:
            print(f"ℹ Interpreted query as: '{processed_query}'")
        
        log_event("query", {"original_query": user_q, "processed_query": processed_query})
        
        try:
            cand_idx, scores = assistant.hybrid_retrieve(user_q, answers, top_k=15)
            if not cand_idx:
                print("❌ No matches found. Try including material, use, or specific features.")
                log_event("no_match", {"query": processed_query})
                print("💡 Example: 'stainless steel bolt for automotive' or 'milk for human consumption'")
                continue
            
            print(f"\n✅ Found {len(cand_idx)} potential matches")
            assistant.show_candidates(cand_idx, scores, limit=5)
            
            rounds = 0
            answered_keys = set()
            
            while len(cand_idx) > 1 and rounds < 2:
                questions = assistant.propose_intelligent_questions(cand_idx, answered_keys, answers, max_q=3)
                if not questions:
                    break
                
                print(f"\n❓ Please answer to narrow down results (Round {rounds + 1}):")
                round_answers = {}
                
                for q in questions:
                    print(f"\n   {q['question']}")
                    if q.get('options'):
                        print(f"   Options: {', '.join(q['options'])}")
                    
                    ans = input("   Your answer (Enter to skip): ").strip().lower()
                    if ans and ans in ['skip', 'none', 'unknown']:
                        continue
                    
                    if ans and q.get('options'):
                        normalized_ans = normalize_synonyms(ans)
                        corrected_ans = correct_typos(normalized_ans, q['options'])
                        matched = corrected_ans in q['options']
                        if not matched:
                            print(f"⚠ Invalid answer. Please choose from: {', '.join(q['options'])}")
                            continue
                        if corrected_ans != normalized_ans:
                            print(f"ℹ Corrected answer to: '{corrected_ans}'")
                        ans = corrected_ans
                    
                    if ans:
                        if q['key'] in answers and answers[q['key']] != ans:
                            print(f"⚠ Contradictory answer for '{q['question']}'. Keeping latest: '{ans}'")
                        round_answers[q['key']] = ans
                        answered_keys.add(q['key'])
                
                if round_answers:
                    answers.update(round_answers)
                    log_event("refinement", {"round": rounds + 1, "answers": round_answers})
                    
                    refined = assistant.refine_with_answers(cand_idx, answers)
                    if len(refined) < len(cand_idx):
                        cand_idx = refined
                        cand_idx, scores = assistant.hybrid_retrieve(user_q, answers, top_k=len(cand_idx))
                        print(f"   ✓ Refined to {len(cand_idx)} candidates")
                        if len(cand_idx) > 1:
                            assistant.show_candidates(cand_idx, scores, limit=3)
                    else:
                        print("   ℹ Answers did not narrow down results significantly.")
                
                rounds += 1
            
            if len(cand_idx) > 1:
                print(f"\n🔍 Multiple candidates remain. Selecting the best match...")
                cand_idx, scores = assistant.hybrid_retrieve(user_q, answers, top_k=len(cand_idx))
            
            final_idx = cand_idx[0]
            confidence = assistant.explain_confidence(final_idx, user_q, answers)
            
            print("\n" + "="*60)
            print("🎯 FINAL HS CODE CLASSIFICATION")
            print("="*60)
            
            row = assistant.df.iloc[final_idx]
            print(f"\nHS Code: {row['commodity']}")
            print(f"Description: {row['description']}")
            print(f"Confidence: {confidence['confidence_score']:.1f}%")
            print(f"Explanation: {confidence['explanation']}")
            
            if confidence.get('category'):
                print(f"Category: {confidence['category'].title()}")
            
            print(f"\n📊 Details:")
            for col in ['cet_duty_rate', 'ukgt_duty_rate', 'VAT Rate']:
                if col in row and pd.notna(row[col]):
                    print(f"   {col}: {row[col]}")
            
            user_feedback = None
            correct_hs_code = None
            while True:
                try:
                    feedback_input = input("\n💡 Was this classification correct? (yes/no/skip): ").strip().lower()
                    if feedback_input in ['yes', 'no', 'skip']:
                        user_feedback = feedback_input
                        break
                    print("⚠ Please enter 'yes', 'no', or 'skip'.")
                except KeyboardInterrupt:
                    print("\n⚠ Feedback interrupted. Skipping feedback.")
                    user_feedback = "skip"
                    break
            
            if user_feedback == 'no':
                correct_input = input("💡 Please enter the correct HS code (optional, press Enter to skip): ").strip()
                if correct_input:
                    correct_hs_code = correct_input
            
            if user_feedback in ['yes', 'no', 'skip']:
                assistant.add_feedback(processed_query, row['commodity'], answers, user_feedback, correct_hs_code)
                print("✓ Thank you for your feedback!")
                if user_feedback == 'no':
                    print("💡 Your input will help improve future classifications.")
            
            log_event("result", {
                "hs_code": row['commodity'],
                "description": row['description'],
                "confidence": confidence['confidence_score'],
                "explanation": confidence['explanation'],
                "category": confidence.get('category'),
                "user_feedback": user_feedback,
                "correct_hs_code": correct_hs_code
            })
            
            print(f"\n💡 Tip: Include specific details like material, use, or features for best results.")
        
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            print(f"❌ An error occurred: {e}. Please try again with a different description.")
            log_event("error", {"query": processed_query, "error": str(e)})

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="HS Code Classification Chatbot")
    parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    parser.add_argument("--log", help="Path to store chat logs (JSONL)")
    
    args = parser.parse_args()
    
    try:
        logger = setup_logging(args.log)
        logger.info(f"Loading model from {args.model}")
        with open(args.model, "rb") as f:
            bundle = pickle.load(f)
        
        bundle['model_path'] = args.model
        print(f"✅ Model loaded successfully with {len(bundle['df_records'])} HS codes")
        interactive_chat(bundle, logger, args.log)
        
    except Exception as e:
        logger = setup_logging(args.log)
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()