import numpy as np
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from linguistic_features import extract_linguistic_features


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
        )
        self.linguistic_scaler = StandardScaler()
        self.semantic_scaler = StandardScaler()
        self.semantic_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def fit(self, X, y=None):
        X = list(X)
        self.tfidf.fit(X)

        linguistic_features = np.array(
            [extract_linguistic_features(text) for text in X]
        )
        self.linguistic_scaler.fit(linguistic_features)

        semantic_features = self.semantic_model.encode(
            X,
            batch_size=32,
            show_progress_bar=True,
        )
        self.semantic_scaler.fit(semantic_features)
        return self

    def transform(self, X):
        X = list(X)
        tfidf_features = self.tfidf.transform(X)

        linguistic_features = np.array(
            [extract_linguistic_features(text) for text in X]
        )
        linguistic_features = self.linguistic_scaler.transform(
            linguistic_features
        )

        semantic_features = self.semantic_model.encode(
            X,
            batch_size=32,
            show_progress_bar=True,
        )
        semantic_features = self.semantic_scaler.transform(semantic_features)

        return hstack(
            [tfidf_features, linguistic_features, semantic_features]
        )
