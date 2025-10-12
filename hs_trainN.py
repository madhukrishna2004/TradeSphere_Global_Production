"""
HS CODE TRAINER — Train HS Code Classification Model from Excel

Usage:
  python hs_train.py --excel /path/to/global-uk-tariff.xlsx --out /path/to/tariff_model.pkl

Dependencies: pandas, numpy, scikit-learn, sentence-transformers
"""

import argparse
import re
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import sys
import os

# ---------------------------
# Configuration
# ---------------------------

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HSTrainer")

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
# Utility Functions
# ---------------------------

def normalize_text(s: str) -> str:
    """Normalize text by lowercasing, removing special characters, and collapsing spaces."""
    if s is None or pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """Tokenize text, removing stopwords and short words."""
    if not s:
        return []
    return [w for w in normalize_text(s).split() if w not in STOPWORDS and len(w) > 2]

def normalize_synonyms(text: str) -> str:
    """Replace synonyms with standardized terms."""
    if not text:
        return ""
    words = text.split()
    normalized_words = [SYNONYMS.get(word, word) for word in words]
    return ' '.join(normalized_words)

def extract_facets(desc: str) -> Dict[str, Any]:
    """Extract domain-specific facets from description."""
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
        "food_related": bool(re.search(r"\b(food|milk|cream|butter|cheese|dairy|meat|fruit|vegetable|grain|beverage)\b", d)),
        "mechanical_related": bool(re.search(r"\b(engine|motor|gear|bearing|valve|pump|compressor|lawn mower)\b", d)),
        "electrical_related": bool(re.search(r"\b(voltage|current|circuit|transformer|capacitor|resistor|computer)\b", d)),
        "textile_related": bool(re.search(r"\b(fabric|textile|yarn|fiber|cloth|woven|knitted)\b", d))
    }
    return facets

def detect_category(facets: Dict[str, Any]) -> Optional[str]:
    """Detect the primary category based on facets."""
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
    """Extract required columns, validating presence."""
    cols = [
        "commodity", "description", "cet_duty_rate", "ukgt_duty_rate", "change",
        "trade_remedy_applies", "cet_applies_until_trade_remedy_transition_reviews_concluded",
        "suspension_applies", "atq_applies", "Product-specific rule of origin", "VAT Rate",
        "Product-specific rule of origin japan"
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}. Filling with NA.")
        for col in missing:
            df[col] = pd.NA
    return df[cols].copy()

# ---------------------------
# Training
# ---------------------------

def train_model(excel_path: str, out_pkl: str) -> str:
    """Train HS code classification model and save as a dictionary."""
    try:
        logger.info(f"Loading Excel data from {excel_path}")
        df_raw = pd.read_excel(excel_path, sheet_name="Sheet1")
        df = df_required_subset(df_raw)
        df = df.copy()
        df["commodity"] = df["commodity"].astype(str)
        df["description"] = df["description"].astype(str).fillna("")

        # Validate key entries
        required_codes = {
            '0401': 'milk',
            '8433': 'lawn mower',
            '7318': 'screws',
            '8471': 'computer'
        }
        missing_codes = []
        for code, item in required_codes.items():
            if not df['commodity'].str.startswith(code).any():
                missing_codes.append((code, item))
        if missing_codes:
            logger.error(f"Missing critical HS codes: {', '.join(f'{item} ({code})' for code, item in missing_codes)}. Update Excel file.")
            raise ValueError(f"Missing critical HS codes: {missing_codes}")

        def build_corpus(row):
            return normalize_text(f"{row['commodity']} {row['description']}")

        df["corpus"] = df.apply(build_corpus, axis=1)

        # Create hierarchical information
        df["chapter"] = df["commodity"].str.extract(r'(\d{2})')[0]
        df["heading"] = df["commodity"].str.extract(r'(\d{4})')[0]
        df["subheading"] = df["commodity"].str.extract(r'(\d{6})')[0]

        logger.info("Generating sentence embeddings...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embeddings = embedding_model.encode(
            df["corpus"].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )

        # Precompute hierarchical maps
        logger.info("Building hierarchical maps...")
        chapters_map = defaultdict(list)
        headings_map = defaultdict(list)
        subheadings_map = defaultdict(list)
        category_map = defaultdict(list)

        for i, (chapter, heading, subheading, corpus) in enumerate(zip(df["chapter"], df["heading"], df["subheading"], df["corpus"])):
            if pd.notna(chapter):
                chapters_map[chapter].append(i)
            if pd.notna(heading):
                headings_map[heading].append(i)
            if pd.notna(subheading):
                subheadings_map[subheading].append(i)
            category = detect_category(extract_facets(corpus))
            if category:
                category_map[category].append(i)

        # Build keyword inverted index
        logger.info("Building keyword index...")
        keyword_index = defaultdict(list)
        for i, corpus in enumerate(df["corpus"]):
            tokens = set(tokenize(corpus))
            for token in tokens:
                if len(token) > 2:
                    keyword_index[token].append(i)

        # Convert defaultdict to dict and validate
        chapters_map = dict(chapters_map)
        headings_map = dict(headings_map)
        subheadings_map = dict(subheadings_map)
        keyword_index = dict(keyword_index)
        category_map = dict(category_map)

        # Log index sizes and samples
        logger.info(f"chapters_map: {len(chapters_map)} entries, sample: {list(chapters_map.keys())[:3]}")
        logger.info(f"headings_map: {len(headings_map)} entries, sample: {list(headings_map.keys())[:3]}")
        logger.info(f"subheadings_map: {len(subheadings_map)} entries, sample: {list(subheadings_map.keys())[:3]}")
        logger.info(f"keyword_index: {len(keyword_index)} entries, sample: {list(keyword_index.keys())[:3]}")
        logger.info(f"category_map: {len(category_map)} entries, sample: {list(category_map.keys())}")
        
        # Validate food-related keywords
        food_keywords = ['milk', 'cream', 'dairy']
        for kw in food_keywords:
            if kw not in keyword_index:
                logger.warning(f"Keyword '{kw}' not found in keyword_index. Milk-related queries may fail.")

        # Validate non-empty indexes
        if not keyword_index:
            logger.error("keyword_index is empty. Aborting.")
            raise ValueError("keyword_index is empty.")
        if not chapters_map:
            logger.warning("chapters_map is empty. HS pattern boosting will be limited.")
        if not category_map:
            logger.warning("category_map is empty. Category boosting will be limited.")

        # Create bundle
        bundle = {
            'version': "3.0.0",
            'df_records': df.to_dict(orient="records"),
            'embedding_model': embedding_model,
            'embeddings': embeddings,
            'stopwords': list(STOPWORDS),
            'materials': MATERIALS,
            'uses': USES,
            'features': FEATURES,
            'chapters_map': chapters_map,
            'headings_map': headings_map,
            'subheadings_map': subheadings_map,
            'keyword_index': keyword_index,
            'category_map': category_map
        }

        logger.info(f"Saving model to {out_pkl}")
        with open(out_pkl, "wb") as f:
            pickle.dump(bundle, f)

        logger.info(f"Model contains {len(df)} HS codes")
        return out_pkl

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
    parser.add_argument("--out", default="tariff_modelv2.pkl", help="Output path for model .pkl")
    
    args = parser.parse_args()
    
    try:
        model_path = train_model(args.excel, args.out)
        logger.info(f"Model successfully trained and saved to: {model_path}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        logger.info(f"Model contains: {len(bundle['df_records'])} HS codes")
        logger.info(f"Indexes - keyword_index: {len(bundle['keyword_index'])} entries, "
                    f"chapters_map: {len(bundle['chapters_map'])} entries, "
                    f"category_map: {len(bundle['category_map'])} entries")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if os.name == 'nt':
        os.system('chcp 65001 > nul')  # Set console to UTF-8
    main()