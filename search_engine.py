import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MetadataSearchEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Load embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Create embedding text dynamically from ALL columns
        self.df["embedding_text"] = self.df.apply(
            lambda row: self._build_embedding_text(row), axis=1
        )

        self.embeddings = self.model.encode(
            self.df["embedding_text"].tolist(),
            convert_to_numpy=True
        )

    def _build_embedding_text(self, row):
        text_parts = []

        for col in self.df.columns:
            value = row[col]
            if pd.notna(value):
                text_parts.append(f"{col} is {value}")

        return ". ".join(text_parts)

    def _confidence_label(self, similarity_percent):
        if similarity_percent > 75:
            return "High"
        elif similarity_percent > 55:
            return "Moderate"
        else:
            return "Low"

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:
            similarity_percent = float(similarities[idx]) * 100

            row_data = self.df.iloc[idx].to_dict()
            row_data["similarity"] = similarity_percent
            row_data["confidence"] = self._confidence_label(similarity_percent)

            results.append(row_data)

        return results
