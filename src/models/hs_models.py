# hs_models.py
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

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