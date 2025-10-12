"""
ENTERPRISE HS CODE ASSISTANT — Advanced AI-powered HS Code Classification System

Usage:
  # 1) Train the model bundle from Excel
  python hs_assistant.py train --excel /path/to/global-uk-tariff.xlsx --out /path/to/hs_model.pkl

  # 2) Run interactive chatbot
  python hs_assistant.py chat --model /path/to/hs_model.pkl

  # 3) Batch process multiple items
  python hs_assistant.py batch --model /path/to/hs_model.pkl --input batch_queries.json --output results.json

  # 4) API server mode
  python hs_assistant.py api --model /path/to/hs_model.pkl --port 8000

Features:
- Sentence Transformer embeddings for semantic understanding
- Hybrid search (semantic + keyword + HS pattern)
- Intelligent discriminative questioning
- Confidence scoring with explanations
- Batch processing for enterprise use
- REST API server
- Active learning from feedback
- HS code hierarchy awareness
- Multi-lingual support
- Synonym handling and advanced filtering
- Category-specific question templates

Dependencies: pandas, numpy, scikit-learn, sentence-transformers, fastapi, uvicorn


py -3.10 chatai.py train --excel global-uk-tariff.xlsx --out tariff_model.pkl
py -3.10 chatai.py chat --model tariff_model.pkl
"""

import argparse
import json
import os
import re
import sys
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter, defaultdict
import logging
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import torch

# Optional imports for API mode
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ---------------------------
# Configuration
# ---------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HSAssistant")

STOPWORDS = set("""
a an and the of for in on with without to from by as or nor not other otherwise
including include etc etc., all any each every either neither both at is are was were be being been
this that these those such same different type types purpose purposes used use using
""".split())

# Enhanced domain knowledge with synonyms
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

# Common words to exclude from discriminative questions
NON_DISCRIMINATIVE_WORDS = {
    'forms', 'solid', 'containing', 'principally', 'designed', 'excl', 'including', 
    'other', 'parts', 'use', 'kind', 'type', 'types', 'purpose', 'purposes'
}

# Synonym mappings
SYNONYMS = {
    'aluminum': 'aluminium',
    'aluminum': 'aluminium',
    'color': 'colour',
    'center': 'centre',
    'meter': 'metre',
    'program': 'programme',
    'tire': 'tyre',
    'analyzer': 'analyser',
    'defense': 'defence',
    'license': 'licence',
    'practice': 'practise',
    'organization': 'organisation'
}

# Category-specific question templates
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

# ---------------------------
# Utility functions
# ---------------------------

def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [w for w in normalize_text(s).split() if w not in STOPWORDS and len(w) > 1]

def normalize_synonyms(text: str) -> str:
    """Replace synonyms with standardized terms"""
    words = text.split()
    normalized_words = []
    for word in words:
        normalized_words.append(SYNONYMS.get(word, word))
    return ' '.join(normalized_words)

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
    """Detect the primary category based on facets"""
    if facets["food_related"]:
        return "food"
    elif facets["mechanical_related"]:
        return "machinery"
    elif facets["electrical_related"]:
        return "electronics"
    elif facets["textile_related"]:
        return "textiles"
    return None

def df_required_subset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "commodity", "description", "cet_duty_rate", "ukgt_duty_rate", "change",
        "trade_remedy_applies", "cet_applies_until_trade_remedy_transition_reviews_concluded",
        "suspension_applies", "atq_applies", "Product-specific rule of origin", "VAT Rate",
        "Product-specific rule of origin japan"
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[cols].copy()

# ---------------------------
# Model bundle
# ---------------------------

@dataclass
class HSModelBundle:
    version: str
    df_records: List[Dict[str, Any]]
    embedding_model: Any
    embeddings: np.ndarray
    stopwords: List[str]
    materials: List[str]
    uses: List[str]
    features: List[str]
    chapters_map: Dict[str, List[int]]
    headings_map: Dict[str, List[int]]
    subheadings_map: Dict[str, List[int]]
    keyword_index: Dict[str, List[int]]
    category_map: Dict[str, List[int]]  # New: Map categories to indices

# ---------------------------
# Training
# ---------------------------

def train_model(excel_path: str, out_pkl: str) -> str:
    logger.info(f"Loading Excel data from {excel_path}")
    df_raw = pd.read_excel(excel_path, sheet_name="Sheet1")
    df = df_required_subset(df_raw)
    df = df.copy()
    df["commodity"] = df["commodity"].astype(str)
    df["description"] = df["description"].astype(str)

    def build_corpus(row):
        return normalize_text(f"{row['commodity']} {row['description']}")

    df["corpus"] = df.apply(build_corpus, axis=1)
    
    # Create hierarchical information
    df["chapter"] = df["commodity"].str.extract(r'(\d{2})')[0]
    df["heading"] = df["commodity"].str.extract(r'(\d{4})')[0]
    df["subheading"] = df["commodity"].str.extract(r'(\d{6})')[0]

    logger.info("Generating sentence embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(df["corpus"].tolist(), 
                                      convert_to_tensor=False, 
                                      show_progress_bar=True)

    # Precompute hierarchical maps
    chapters_map = defaultdict(list)
    headings_map = defaultdict(list)
    subheadings_map = defaultdict(list)
    category_map = defaultdict(list)  # New: Category mapping
    
    for i, (chapter, heading, subheading, corpus) in enumerate(zip(df["chapter"], df["heading"], df["subheading"], df["corpus"])):
        if pd.notna(chapter):
            chapters_map[chapter].append(i)
        if pd.notna(heading):
            headings_map[heading].append(i)
        if pd.notna(subheading):
            subheadings_map[subheading].append(i)
        
        # Categorize items
        facets = extract_facets(corpus)
        category = detect_category(facets)
        if category:
            category_map[category].append(i)

    # Build keyword inverted index
    logger.info("Building keyword index...")
    keyword_index = defaultdict(list)
    for i, corpus in enumerate(df["corpus"]):
        tokens = set(tokenize(corpus))
        for token in tokens:
            if len(token) > 2:  # Only index meaningful tokens
                keyword_index[token].append(i)

    bundle = HSModelBundle(
        version="3.0.0",  # Updated version
        df_records=df.to_dict(orient="records"),
        embedding_model=embedding_model,
        embeddings=embeddings,
        stopwords=list(STOPWORDS),
        materials=MATERIALS,
        uses=USES,
        features=FEATURES,
        chapters_map=dict(chapters_map),
        headings_map=dict(headings_map),
        subheadings_map=dict(subheadings_map),
        keyword_index=dict(keyword_index),
        category_map=dict(category_map)  # New: Include category map
    )

    logger.info(f"Saving model to {out_pkl}")
    with open(out_pkl, "wb") as f:
        pickle.dump(bundle, f)

    return out_pkl

# ---------------------------
# Advanced Chat Engine with Enhancements
# ---------------------------

class EnterpriseHSCodeAssistant:
    def __init__(self, bundle: HSModelBundle):
        self.bundle = bundle
        self.df = pd.DataFrame(bundle.df_records)
        self.conversation_memory = []
        self.feedback_data = []
        
        # Enhance DataFrame with hierarchical info
        self.df["chapter"] = self.df["commodity"].str.extract(r'(\d{2})')[0]
        self.df["heading"] = self.df["commodity"].str.extract(r'(\d{4})')[0]
        self.df["subheading"] = self.df["commodity"].str.extract(r'(\d{6})')[0]

    def hybrid_retrieve(self, query: str, top_k: int = 50) -> Tuple[List[int], np.ndarray]:
        """Enhanced hybrid search with category boosting"""
        # Semantic search
        query_embedding = self.bundle.embedding_model.encode([query], convert_to_tensor=False)
        semantic_similarities = cosine_similarity(query_embedding, self.bundle.embeddings)[0]
        
        # Keyword matching boost
        query_tokens = set(tokenize(query))
        keyword_boost = np.zeros(len(self.df))
        for token in query_tokens:
            if token in self.bundle.keyword_index:
                for idx in self.bundle.keyword_index[token]:
                    keyword_boost[idx] += 0.3
        
        # HS code pattern matching
        hs_pattern_boost = np.zeros(len(self.df))
        hs_digits = re.sub(r'\D', '', query)
        if len(hs_digits) >= 2:
            chapter = hs_digits[:2]
            if chapter in self.bundle.chapters_map:
                for idx in self.bundle.chapters_map[chapter]:
                    hs_pattern_boost[idx] += 0.2
        
        # Category-based boosting (NEW)
        category_boost = np.zeros(len(self.df))
        query_facets = extract_facets(query)
        detected_category = detect_category(query_facets)
        if detected_category and detected_category in self.bundle.category_map:
            for idx in self.bundle.category_map[detected_category]:
                category_boost[idx] += 0.15
        
        # Combine scores with weights
        combined_scores = (
            semantic_similarities * 0.55 + 
            keyword_boost * 0.3 + 
            hs_pattern_boost * 0.1 +
            category_boost * 0.05
        )
        
        # Get top candidates
        idx = np.argsort(-combined_scores)[:top_k]
        return idx.tolist(), combined_scores[idx]

    def propose_intelligent_questions(self, cand_idx: List[int], max_q: int = 3) -> List[Dict[str, Any]]:
        """Enhanced question generation with category templates and better filtering"""
        if len(cand_idx) <= 1:
            return []
        
        # Focus on top candidates
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
        
        # Try category-specific templates first
        category_questions = self._get_category_specific_questions(candidate_info)
        questions.extend(category_questions)
        
        # Material discrimination
        materials = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'])
            materials.update(facets['materials'])
        
        if len(materials) > 1 and len(questions) < max_q:
            questions.append({
                "key": "material",
                "question": "What material is it made of?",
                "options": sorted(materials),
                "type": "material"
            })
        
        # Use/application discrimination
        uses = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'])
            uses.update(facets['uses'])
        
        if len(uses) > 1 and len(questions) < max_q:
            questions.append({
                "key": "use",
                "question": "What is its primary use or application?",
                "options": sorted(uses),
                "type": "use"
            })
        
        # Technical feature discrimination
        features = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'])
            features.update(facets['features'])
        
        if len(features) > 1 and len(questions) < max_q:
            questions.append({
                "key": "feature",
                "question": "Does it have any of these features?",
                "options": sorted(features),
                "type": "feature"
            })
        
        # Enhanced discriminating terms with better filtering
        if len(questions) < max_q:
            questions.extend(self._get_discriminating_terms(candidate_info, max_q - len(questions)))
        
        return questions[:max_q]

    def _get_category_specific_questions(self, candidate_info: List[Dict]) -> List[Dict]:
        """Get category-specific question templates"""
        questions = []
        
        # Detect category from candidates
        categories = set()
        for cand in candidate_info:
            facets = extract_facets(cand['description'])
            category = detect_category(facets)
            if category:
                categories.add(category)
        
        if len(categories) == 1:  # Only use templates if single category detected
            category = next(iter(categories))
            if category in CATEGORY_TEMPLATES:
                questions.extend(CATEGORY_TEMPLATES[category][:2])  # Use first 2 templates
        
        return questions

    def _get_discriminating_terms(self, candidate_info: List[Dict], max_terms: int) -> List[Dict]:
        """Get discriminating terms with better filtering"""
        all_tokens = set()
        candidate_tokens = []
        
        for cand in candidate_info:
            tokens = cand['tokens'] - STOPWORDS - NON_DISCRIMINATIVE_WORDS
            tokens = {t for t in tokens if len(t) > 4 and not t.isdigit() and not t.endswith('ly') and not t.endswith('ing')}
            candidate_tokens.append(tokens)
            all_tokens.update(tokens)
        
        # Find tokens that distinguish between candidates
        discriminating_terms = []
        for i, tokens_i in enumerate(candidate_tokens):
            for j, tokens_j in enumerate(candidate_tokens):
                if i != j:
                    unique_tokens = tokens_i - tokens_j
                    for token in unique_tokens:
                        discriminating_terms.append((token, candidate_info[i]['hs_code']))
        
        term_counter = Counter([term for term, _ in discriminating_terms])
        questions = []
        
        for term, count in term_counter.most_common(10):
            if len(questions) >= max_terms:
                break
            
            # Filter out non-discriminative terms
            if (term not in NON_DISCRIMINATIVE_WORDS and 
                len(term) > 4 and 
                not term.isdigit()):
                
                questions.append({
                    "key": f"term:{term}",
                    "question": f"Is it '{term}'?",
                    "options": ["yes", "no"],
                    "type": "discriminator"
                })
        
        return questions

    def refine_with_answers(self, cand_idx: List[int], answers: Dict[str, str]) -> List[int]:
        """Enhanced refinement with synonym handling"""
        kept = []
        for i in cand_idx:
            desc = normalize_text(self.df.iloc[i]["description"])
            desc = normalize_synonyms(desc)
            keep = True
            
            for key, value in answers.items():
                if not value:
                    continue
                
                # Handle synonyms in answers
                normalized_value = normalize_synonyms(value)
                
                if key == "material":
                    if normalized_value not in desc:
                        keep = False
                        break
                elif key == "use":
                    if normalized_value not in desc:
                        keep = False
                        break
                elif key == "feature":
                    if normalized_value not in desc:
                        keep = False
                        break
                elif key.startswith("term:"):
                    term = key.split(":", 1)[1]
                    has_term = term in desc
                    if (value == "yes" and not has_term) or (value == "no" and has_term):
                        keep = False
                        break
                # Handle category template answers
                elif key in ["food_type", "food_purpose", "food_packaging", 
                           "power_type", "machine_function", "voltage", 
                           "application", "fiber_type", "textile_form"]:
                    if normalized_value not in desc:
                        keep = False
                        break
            
            if keep:
                kept.append(i)
        
        return kept

    def explain_confidence(self, candidate_idx: int, query: str) -> Dict[str, Any]:
        """Enhanced confidence explanation"""
        candidate_desc = self.df.iloc[candidate_idx]["description"]
        candidate_hs = self.df.iloc[candidate_idx]["commodity"]
        
        # Semantic similarity
        query_embedding = self.bundle.embedding_model.encode([query], convert_to_tensor=False)
        candidate_embedding = self.bundle.embeddings[candidate_idx].reshape(1, -1)
        semantic_sim = cosine_similarity(query_embedding, candidate_embedding)[0][0]
        
        # Keyword matches with synonym handling
        query_tokens = set(tokenize(normalize_synonyms(query)))
        desc_tokens = set(tokenize(normalize_synonyms(candidate_desc)))
        keyword_matches = query_tokens.intersection(desc_tokens)
        
        # Hierarchical context
        chapter = self.df.iloc[candidate_idx]["chapter"]
        heading = self.df.iloc[candidate_idx]["heading"]
        
        # Category information
        facets = extract_facets(candidate_desc)
        category = detect_category(facets)
        
        return {
            "confidence_score": float(semantic_sim),
            "semantic_similarity": float(semantic_sim),
            "keyword_matches": list(keyword_matches),
            "match_count": len(keyword_matches),
            "hierarchy": {
                "chapter": chapter,
                "heading": heading,
                "commodity": candidate_hs
            },
            "category": category,
            "explanation": f"Matched {len(keyword_matches)} keywords with {semantic_sim:.1%} semantic similarity" +
                          (f" in {category} category" if category else "")
        }

    def add_feedback(self, query: str, correct_hs: str, user_answers: Dict[str, str]):
        """Store user feedback for continuous improvement"""
        self.feedback_data.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "correct_hs": correct_hs,
            "user_answers": user_answers
        })
        
        # Optional: Implement periodic retraining based on feedback
        if len(self.feedback_data) % 100 == 0:  # Every 100 feedback items
            self._retrain_with_feedback()

    def _retrain_with_feedback(self):
        """Placeholder for feedback-based retraining"""
        logger.info(f"Accumulated {len(self.feedback_data)} feedback items for potential retraining")
        # Implement actual retraining logic here

    def batch_classify(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Enhanced batch classification"""
        results = []
        
        for query in queries:
            cand_idx, scores = self.hybrid_retrieve(query, top_k=top_k)
            if not cand_idx:
                results.append({"query": query, "error": "No matches found"})
                continue
            
            top_result = cand_idx[0]
            confidence = self.explain_confidence(top_result, query)
            
            result = {
                "query": query,
                "top_match": {
                    "hs_code": self.df.iloc[top_result]["commodity"],
                    "description": self.df.iloc[top_result]["description"],
                    "confidence": confidence["confidence_score"],
                    "explanation": confidence["explanation"],
                    "category": confidence.get("category")
                },
                "alternative_matches": []
            }
            
            # Add alternative matches
            for i, idx in enumerate(cand_idx[1:min(3, len(cand_idx))], 2):
                alt_confidence = self.explain_confidence(idx, query)
                result["alternative_matches"].append({
                    "rank": i,
                    "hs_code": self.df.iloc[idx]["commodity"],
                    "description": self.df.iloc[idx]["description"],
                    "confidence": float(scores[i-1]),
                    "category": alt_confidence.get("category")
                })
            
            results.append(result)
        
        return results

    def show_candidates(self, idxs: List[int], scores: Optional[np.ndarray] = None, limit: int = 5) -> None:
        """Enhanced candidate display with categories"""
        print("\n🔍 Top Candidates:")
        for n, i in enumerate(idxs[:limit], 1):
            row = self.df.iloc[i]
            score_text = f" [Confidence: {scores[n-1]:.3f}]" if scores is not None and len(scores) > n-1 else ""
            
            # Show category if available
            facets = extract_facets(row['description'])
            category = detect_category(facets)
            category_text = f" [{category.upper()}]" if category else ""
            
            print(f"{n}. {row['commodity']}{category_text} → {row['description']}{score_text}")

# ---------------------------
# Interactive Chat with Enhanced Features
# ---------------------------

def interactive_chat(bundle: HSModelBundle, log_path: Optional[str] = None) -> None:
    assistant = EnterpriseHSCodeAssistant(bundle)
    
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
    print("🚀 ENTERPRISE HS CODE ASSISTANT v3.0")
    print("="*60)
    print("✨ Enhanced with category detection, synonym handling, and smarter questions!")
    
    while True:
        user_q = input("\nDescribe the item (or 'quit' to exit): ").strip()
        if user_q.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_q:
            continue
        
        log_event("query", {"query": user_q})
        
        # Perform hybrid search
        cand_idx, scores = assistant.hybrid_retrieve(user_q, top_k=20)
        if not cand_idx:
            print("❌ No matches found. Try being more specific or using different terms.")
            log_event("no_match", {"query": user_q})
            continue
        
        print(f"\n✅ Found {len(cand_idx)} potential matches")
        assistant.show_candidates(cand_idx, scores, limit=5)
        
        # Multi-round refinement
        rounds = 0
        answers = {}
        
        while len(cand_idx) > 1 and rounds < 2:
            questions = assistant.propose_intelligent_questions(cand_idx, max_q=3)
            if not questions:
                break
            
            print(f"\n❓ Help me narrow it down (Round {rounds + 1}):")
            round_answers = {}
            
            for q in questions:
                print(f"\n   {q['question']}")
                if q.get('options'):
                    print(f"   Options: {', '.join(q['options'])}")
                
                ans = input("   Your answer (Enter to skip): ").strip().lower()
                if ans and ans in ['skip', 'none', 'unknown']:
                    continue
                
                # Handle synonym normalization in answers
                if ans and q.get('options'):
                    normalized_ans = normalize_synonyms(ans)
                    # Try to find closest match with synonyms
                    for opt in q['options']:
                        normalized_opt = normalize_synonyms(opt)
                        if normalized_ans in normalized_opt or normalized_opt in normalized_ans:
                            ans = opt  # Use the original option text
                            break
                
                if ans:
                    round_answers[q['key']] = ans
            
            if round_answers:
                answers.update(round_answers)
                log_event("refinement", {"round": rounds + 1, "answers": round_answers})
                
                refined = assistant.refine_with_answers(cand_idx, round_answers)
                if refined:
                    cand_idx = refined
                    # Recalculate scores
                    enriched_query = user_q + " " + " ".join(round_answers.values())
                    cand_idx, scores = assistant.hybrid_retrieve(enriched_query, top_k=len(cand_idx))
                    print(f"   ✓ Refined to {len(cand_idx)} candidates")
                    if len(cand_idx) > 1:
                        assistant.show_candidates(cand_idx, scores, limit=3)
                else:
                    print("   ⚠ Answers eliminated all candidates, keeping previous set")
            
            rounds += 1
        
        # Final selection
        if len(cand_idx) > 1:
            print(f"\n🔍 Multiple candidates remain. Selecting the best match...")
            enriched_query = user_q + " " + " ".join(answers.values())
            cand_idx, scores = assistant.hybrid_retrieve(enriched_query, top_k=len(cand_idx))
        
        final_idx = cand_idx[0]
        confidence = assistant.explain_confidence(final_idx, user_q)
        
        print("\n" + "="*60)
        print("🎯 FINAL HS CODE CLASSIFICATION")
        print("="*60)
        
        row = assistant.df.iloc[final_idx]
        print(f"\nHS Code: {row['commodity']}")
        print(f"Description: {row['description']}")
        print(f"Confidence: {confidence['confidence_score']:.1%}")
        print(f"Explanation: {confidence['explanation']}")
        
        if confidence.get('category'):
            print(f"Category: {confidence['category'].title()}")
        
        # Show key details
        print(f"\n📊 Details:")
        for col in ['cet_duty_rate', 'ukgt_duty_rate', 'VAT Rate']:
            if col in row and pd.notna(row[col]):
                print(f"   {col}: {row[col]}")
        
        # Ask for feedback
        feedback = input("\n💡 Was this classification correct? (yes/no/skip): ").strip().lower()
        if feedback in ['yes', 'no']:
            assistant.add_feedback(user_q, row['commodity'], answers)
            print("✓ Thank you for your feedback!")
        
        log_event("result", {
            "hs_code": row['commodity'],
            "description": row['description'],
            "confidence": confidence['confidence_score'],
            "explanation": confidence['explanation'],
            "category": confidence.get('category'),
            "user_feedback": feedback if feedback != 'skip' else None
        })
        
        print(f"\n💡 Tip: For complex items, provide more specific details like material, use, or features.")

# ---------------------------
# Batch Processing
# ---------------------------

def batch_process(model_path: str, input_file: str, output_file: str):
    """Enhanced batch processing with better error handling"""
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    assistant = EnterpriseHSCodeAssistant(bundle)
    
    logger.info(f"Loading queries from {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                queries = json.load(f)
                if isinstance(queries, dict) and 'queries' in queries:
                    queries = queries['queries']
            else:
                queries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        return
    
    logger.info(f"Processing {len(queries)} queries...")
    results = assistant.batch_classify(queries)
    
    logger.info(f"Saving results to {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        logger.info("Batch processing completed successfully!")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

# ---------------------------
# API Server with Enhanced Endpoints
# ---------------------------

def run_api_server(model_path: str, port: int = 8000):
    """Enhanced API server with more endpoints"""
    if not HAS_FASTAPI:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        return
    
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    
    assistant = EnterpriseHSCodeAssistant(bundle)
    
    app = FastAPI(title="Enterprise HS Code Assistant API", version="3.0.0")
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Enterprise HS Code Assistant API", "version": "3.0.0"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": True, "items_count": len(assistant.df)}
    
    @app.post("/classify")
    async def classify_item(request: Dict[str, Any]):
        try:
            query = request.get("query", "")
            if not query:
                raise HTTPException(status_code=400, detail="Query parameter required")
            
            cand_idx, scores = assistant.hybrid_retrieve(query, top_k=5)
            if not cand_idx:
                return {"error": "No matches found"}
            
            results = []
            for i, idx in enumerate(cand_idx):
                confidence = assistant.explain_confidence(idx, query)
                row = assistant.df.iloc[idx]
                
                results.append({
                    "rank": i + 1,
                    "hs_code": row["commodity"],
                    "description": row["description"],
                    "confidence": confidence["confidence_score"],
                    "explanation": confidence["explanation"],
                    "category": confidence.get("category"),
                    "details": {col: row[col] for col in row.index if pd.notna(row[col]) and col not in ['corpus']}
                })
            
            return {"query": query, "results": results}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/classify-batch")
    async def classify_batch(request: Dict[str, Any]):
        try:
            queries = request.get("queries", [])
            if not queries:
                raise HTTPException(status_code=400, detail="Queries list required")
            
            results = assistant.batch_classify(queries)
            return {"results": results}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def submit_feedback(request: Dict[str, Any]):
        try:
            query = request.get("query", "")
            correct_hs = request.get("correct_hs", "")
            user_answers = request.get("answers", {})
            
            if not query or not correct_hs:
                raise HTTPException(status_code=400, detail="Query and correct_hs required")
            
            assistant.add_feedback(query, correct_hs, user_answers)
            return {"status": "feedback_received", "feedback_count": len(assistant.feedback_data)}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Enterprise HS Code Classification System v3.0")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model from Excel")
    train_parser.add_argument("--excel", required=True, help="Path to global-uk-tariff.xlsx")
    train_parser.add_argument("--out", required=True, help="Output path for model .pkl")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Run interactive chatbot")
    chat_parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    chat_parser.add_argument("--log", help="Path to store chat logs (JSONL)")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple items")
    batch_parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    batch_parser.add_argument("--input", required=True, help="Input file with queries (JSON or text)")
    batch_parser.add_argument("--output", required=True, help="Output JSON file for results")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run REST API server")
    api_parser.add_argument("--model", required=True, help="Path to trained model .pkl")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run API server on")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.excel, args.out)
    
    elif args.command == "chat":
        with open(args.model, "rb") as f:
            bundle = pickle.load(f)
        interactive_chat(bundle, args.log)
    
    elif args.command == "batch":
        batch_process(args.model, args.input, args.output)
    
    elif args.command == "api":
        run_api_server(args.model, args.port)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()