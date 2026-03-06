import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MetadataSearchEngine:

    def __init__(self, csv_path="talk2acquisition_master_metadata_v2.csv"):
        print("Loading metadata dataset...")

        self.df = pd.read_csv(csv_path)

        self.df.fillna("", inplace=True)

        print("Loading AI model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Generating embeddings...")

        self.df["combined_text"] = (
            self.df["attribute_name"].astype(str) + " " +
            self.df["definition"].astype(str) + " " +
            self.df["synonyms"].astype(str)
        )

        self.embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            convert_to_numpy=True
        )

        print("Embeddings ready.")

    def get_confidence(self, score):

        if score > 0.75:
            return "Very High"
        elif score > 0.60:
            return "High"
        elif score > 0.45:
            return "Moderate"
        elif score > 0.30:
            return "Low"
        else:
            return "Very Low"

    def search(self, query, top_k=5):

        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = self.df.iloc[top_indices].copy()

        results["similarity"] = similarities[top_indices]

        results["confidence"] = results["similarity"].apply(self.get_confidence)

        return results.reset_index(drop=True)
