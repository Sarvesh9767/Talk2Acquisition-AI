import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MetadataSearchEngine:
    def __init__(self, csv_path):
        # Load dataset
        self.df = pd.read_csv(csv_path)
        self.df = self.df.fillna("")

        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Build natural-language embedding text
        self.df["combined_text"] = self.df.apply(self._build_embedding_text, axis=1)

        # Generate normalized embeddings
        self.embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            normalize_embeddings=True
        )

    def _build_embedding_text(self, row):
        """
        Create natural language sentences for stronger semantic matching.
        This dramatically improves governance and frequency queries.
        """
        return f"""
        {row['attribute_name']} is defined as {row['definition']}.
        It belongs to the {row['asset_class']} asset class and is provided by {row['vendor']}.
        The update frequency of {row['attribute_name']} is {row['frequency']}.
        The business owner of {row['attribute_name']} is {row['business_owner']}.
        The data steward responsible for {row['attribute_name']} is {row['data_steward']}.
        The regulatory source governing {row['attribute_name']} is {row['regulatory_source']}.
        Synonyms for {row['attribute_name']} include {row['synonyms']}.
        """

    def search(self, query, top_k=5):
        """
        Perform semantic search on metadata.
        Returns top_k results.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:
            score = similarities[idx]
            similarity_percent = round(float(score) * 100, 2)

            results.append({
                "attribute_name": self.df.iloc[idx]["attribute_name"],
                "vendor": self.df.iloc[idx]["vendor"],
                "asset_class": self.df.iloc[idx]["asset_class"],
                "definition": self.df.iloc[idx]["definition"],
                "frequency": self.df.iloc[idx]["frequency"],
                "business_owner": self.df.iloc[idx]["business_owner"],
                "data_steward": self.df.iloc[idx]["data_steward"],
                "regulatory_source": self.df.iloc[idx]["regulatory_source"],
                "similarity": similarity_percent,
                "confidence": self._confidence_label(similarity_percent)
            })

        return results

    def _confidence_label(self, similarity_percent):
        """
        Convert similarity score to confidence label.
        """
        if similarity_percent >= 75:
            return "Very High"
        elif similarity_percent >= 60:
            return "High"
        elif similarity_percent >= 45:
            return "Moderate"
        else:
            return "Low"
