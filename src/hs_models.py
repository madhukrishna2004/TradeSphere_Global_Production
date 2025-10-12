# src/hs_models.py
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os

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
    category_map: Dict[str, List[int]]

    @classmethod
    def from_data_file(cls, data_path: str, model_name: str = "all-MiniLM-L6-v2", version: str = "1.0.0") -> 'HSModelBundle':
        """
        Initialize HSModelBundle from a data file (pickle or Excel).
        
        Args:
            data_path (str): Path to HS code data (pickle or Excel).
            model_name (str): SentenceTransformer model name.
            version (str): Model version.
        
        Returns:
            HSModelBundle: Initialized model bundle.
        """
        # Load HS code data
        try:
            if data_path.endswith('.pkl'):
                df = pd.read_pickle(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Data file must be .pkl or .xlsx")
            df_records = df.to_dict(orient="records")
        except Exception as e:
            raise ValueError(f"Failed to load data from {data_path}: {e}")

        # Initialize embedding model
        embedding_model = SentenceTransformer(model_name)
        
        # Generate embeddings for descriptions
        descriptions = [record["description"] for record in df_records]
        embeddings = embedding_model.encode(descriptions, convert_to_tensor=False)
        
        # Define domain knowledge (must match chatbot_routes.py)
        stopwords = [
            "a", "an", "and", "the", "of", "for", "in", "on", "with", "without", "to", "from", "by",
            "as", "or", "nor", "not", "other", "otherwise", "including", "include", "etc", "etc.,",
            "all", "any", "each", "every", "either", "neither", "both", "at", "is", "are", "was",
            "were", "be", "being", "been", "this", "that", "these", "those", "such", "same",
            "different", "type", "types", "purpose", "purposes", "used", "use", "using"
        ]
        materials = [
            "steel", "stainless steel", "stainless", "iron", "cast iron", "copper", "aluminium",
            "aluminum", "zinc", "nickel", "titanium", "magnesium", "lead", "tin", "brass", "bronze",
            "plastic", "polymer", "rubber", "wood", "paper", "cardboard", "ceramic", "glass",
            "textile", "leather", "fabric", "stone", "cement", "concrete", "gold", "silver",
            "platinum", "palladium", "composite", "alloy"
        ]
        uses = [
            "automotive", "motor vehicle", "vehicle", "car", "truck", "railway", "aircraft",
            "aerospace", "marine", "ship", "boat", "construction", "building", "infrastructure",
            "furniture", "electrical", "electronics", "telecom", "medical", "surgical",
            "pharmaceutical", "agriculture", "mining", "oil", "gas", "food", "beverage",
            "packaging", "machine tools", "hand tools", "household", "industrial", "office",
            "school", "laboratory", "HVAC", "plumbing", "sanitary", "solar", "battery",
            "printing", "computing", "telecommunication", "audio", "video", "optical",
            "consumer", "commercial", "residential"
        ]
        features = [
            "threaded", "non-threaded", "coated", "plated", "galvanised", "galvanized",
            "zinc plated", "cold-rolled", "hot-rolled", "forged", "cast", "welded",
            "seamless", "alloy", "non-alloy", "refined", "unrefined", "polished",
            "anodized", "painted", "lacquered", "insulated", "waterproof", "fireproof",
            "antistatic", "magnetic", "non-magnetic", "self-tapping", "hexagonal"
        ]
        
        # Create keyword index
        keyword_index = defaultdict(list)
        for idx, record in enumerate(df_records):
            desc = record["description"].lower()
            tokens = set(desc.split())
            for token in tokens:
                keyword_index[token].append(idx)
        
        # Create chapters, headings, and subheadings maps
        chapters_map = defaultdict(list)
        headings_map = defaultdict(list)
        subheadings_map = defaultdict(list)
        for idx, record in enumerate(df_records):
            commodity = str(record["commodity"])
            if len(commodity) >= 2:
                chapters_map[commodity[:2]].append(idx)
            if len(commodity) >= 4:
                headings_map[commodity[:4]].append(idx)
            if len(commodity) >= 6:
                subheadings_map[commodity[:6]].append(idx)
        
        # Create category map
        category_map = defaultdict(list)
        for idx, record in enumerate(df_records):
            category = record.get("category", "general")
            category_map[category].append(idx)
        
        return cls(
            version=version,
            df_records=df_records,
            embedding_model=embedding_model,
            embeddings=embeddings,
            stopwords=stopwords,
            materials=materials,
            uses=uses,
            features=features,
            chapters_map=chapters_map,
            headings_map=headings_map,
            subheadings_map=subheadings_map,
            keyword_index=keyword_index,
            category_map=category_map
        )