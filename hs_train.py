import argparse
import re
import json
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import sys
import os
from pathlib import Path
import gzip

# ---------------------------
# Configuration
# ---------------------------

def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("HSTrainer")

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
    return df[["commodity", "description", "cet_duty_rate", "ukgt_duty_rate"]].copy()  # Keep only essential columns

# ---------------------------
# Training
# ---------------------------

def preprocess_feedback(feedback_path: str, logger: logging.Logger) -> Dict[str, int]:
    """Preprocess feedback to boost correct HS codes."""
    feedback_weights = defaultdict(int)
    if os.path.exists(feedback_path):
        with open(feedback_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    fb = json.loads(line)
                    if fb.get('user_feedback') == 'yes':
                        feedback_weights[fb['correct_hs']] += 1
                    elif fb.get('user_feedback') == 'no' and 'correct_hs_code' in fb:
                        feedback_weights[fb['correct_hs_code']] += 2
                except json.JSONDecodeError:
                    logger.warning(f"Invalid feedback line: {line.strip()}")
    return feedback_weights

def train_model(excel_path: str, out_pkl: str, logger: logging.Logger, feedback_path: Optional[str] = None) -> str:
    try:
        logger.info(f"Loading Excel data from {excel_path}")
        df_raw = pd.read_excel(excel_path, sheet_name="Sheet1")
        df = df_required_subset(df_raw)
        df["commodity"] = df["commodity"].astype(str)
        df["description"] = df["description"].astype(str)

        def build_corpus(row):
            corpus = normalize_text(f"{row['commodity']} {row['description']}")
            facets = extract_facets(corpus)
            if facets["materials"]:
                corpus += " " + " ".join(f"material:{m}" for m in facets["materials"])
            if facets["uses"]:
                corpus += " " + " ".join(f"use:{u}" for u in facets["uses"])
            if facets["features"]:
                corpus += " " + " ".join(f"feature:{f}" for f in facets["features"])
            return corpus

        # Generate corpus for indexing, but don't store in DataFrame
        corpus_list = [build_corpus(row) for _, row in df.iterrows()]
        
        # Create hierarchical information
        df["chapter"] = df["commodity"].str.extract(r'(\d{2})')[0]
        df["heading"] = df["commodity"].str.extract(r'(\d{4})')[0]
        df["subheading"] = df["commodity"].str.extract(r'(\d{6})')[0]

        logger.info("Generating sentence embeddings...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(
            corpus_list, 
            convert_to_tensor=False, 
            show_progress_bar=True
        ).astype(np.float16)  # Convert to float16

        # Apply feedback weights to embeddings
        if feedback_path:
            logger.info(f"Applying feedback from {feedback_path}")
            feedback_weights = preprocess_feedback(feedback_path, logger)
            for hs_code, weight in feedback_weights.items():
                indices = df.index[df['commodity'] == hs_code].tolist()
                for idx in indices:
                    embeddings[idx] *= (1 + weight * 0.05)  # Boost embeddings for confirmed HS codes

        # Precompute hierarchical maps
        chapters_map = defaultdict(list)
        headings_map = defaultdict(list)
        subheadings_map = defaultdict(list)
        category_map = defaultdict(list)
        
        for i, (chapter, heading, subheading, corpus) in enumerate(zip(df["chapter"], df["heading"], df["subheading"], corpus_list)):
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
        min_freq = 2  # Only keep keywords appearing in at least 2 documents
        for i, corpus in enumerate(corpus_list):
            tokens = set(tokenize(corpus))
            # Include synonyms and facets in index
            for token in tokens:
                if len(token) > 2:
                    keyword_index[token].append(i)
                    for syn_key, syn_val in SYNONYMS.items():
                        if token == syn_key:
                            keyword_index[syn_val].append(i)
                        elif token == syn_val:
                            keyword_index[syn_key].append(i)
            # Index facets
            facets = extract_facets(corpus)
            for material in facets["materials"]:
                keyword_index[f"material:{material}"].append(i)
            for use in facets["uses"]:
                keyword_index[f"use:{use}"].append(i)
            for feature in facets["features"]:
                keyword_index[f"feature:{feature}"].append(i)
        keyword_index = {k: v for k, v in keyword_index.items() if len(v) >= min_freq}

        # Create bundle without embedding_model
        bundle = {
            'version': "4.3.1",
            'df_records': df[["commodity", "description", "cet_duty_rate", "ukgt_duty_rate"]].to_dict(orient="records"),
            'embeddings': embeddings,
            'stopwords': list(STOPWORDS),
            'materials': MATERIALS,
            'uses': USES,
            'features': FEATURES,
            'chapters_map': dict(chapters_map),
            'headings_map': dict(headings_map),
            'subheadings_map': dict(subheadings_map),
            'keyword_index': dict(keyword_index),
            'category_map': dict(category_map),
            'category_templates': CATEGORY_TEMPLATES
        }

        # Save with gzip compression
        logger.info(f"Saving compressed model to {out_pkl}.gz")
        with gzip.open(f"{out_pkl}.gz", "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Model contains {len(df)} HS codes")
        return f"{out_pkl}.gz"

    except FileNotFoundError as e:
        logger.error(f"Excel file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="HS Code Classification Model Trainer")
    parser.add_argument("--excel", required=True, help="Path to global-uk-tariff.xlsx")
    parser.add_argument("--out", required=True, help="Output path for model .pkl")
    parser.add_argument("--feedback", help="Path to feedback.jsonl for retraining", default=None)
    
    args = parser.parse_args()
    
    try:
        log_path = os.path.join(os.path.dirname(args.out), 'train.log')
        logger = setup_logging(log_path)
        model_path = train_model(args.excel, args.out, logger, args.feedback)
        print(f"✅ Model successfully trained and saved to: {model_path}")
        with gzip.open(model_path, "rb") as f:
            bundle = pickle.load(f)
        print(f"📊 Model contains: {len(bundle['df_records'])} HS codes")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()