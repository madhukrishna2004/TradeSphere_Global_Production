import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load Excel file
excel_path = "D:\\ai project\\global-uk-tariffv1\\global-uk-tariff - Copy (2).xlsx"
df = pd.read_excel(excel_path, sheet_name="Sheet1")

# Save DataFrame as pickle
df_pickle_path = "D:\\ai project\\global-uk-tariffv1\\hs_code_data.pkl"
df.to_pickle(df_pickle_path)
print(f"Saved DataFrame to {df_pickle_path}")

# Generate embeddings for FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
descriptions = df['description'].astype(str).tolist()
embeddings = model.encode(descriptions, convert_to_tensor=False)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss_index_path = "D:\\ai project\\global-uk-tariffv1\\hs_code_index.faiss"
faiss.write_index(index, faiss_index_path)
print(f"Saved FAISS index to {faiss_index_path}")