from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RetrievalResult, TextChunk


class LocalVectorRetriever:
    """使用字符 n-gram TF-IDF 的本地向量检索器，适合中英文混合文档。"""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunks: list[TextChunk] = []

    @property
    def ready(self) -> bool:
        return (
            self.vectorizer is not None
            and self.matrix is not None
            and bool(self.chunks)
        )

    def build(self, chunks: list[TextChunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            if self.index_path.exists():
                self.index_path.unlink()
            return
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            chunk.content for chunk in self.chunks
        )
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.ready or not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievalResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in indices
            if scores[index] > 0
        ]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "chunks": [chunk.to_dict() for chunk in self.chunks],
            },
            self.index_path,
        )

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        payload = joblib.load(self.index_path)
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.chunks = [
            TextChunk.from_dict(item) for item in payload["chunks"]
        ]
        return self.ready

