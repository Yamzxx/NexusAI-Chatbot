"""
intent_classifier.py — TF-IDF based intent classification with cosine similarity
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import preprocess


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    intent: str
    confidence: float          # 0.0 to 1.0
    response: str
    source: str               # "intent" | "similarity" | "faq" | "fallback" | "clarify"
    matched_pattern: str = ""
    latency_ms: float = 0.0


class IntentClassifier:
    """
    TF-IDF + cosine similarity intent classifier.

    Loads intent patterns from intents.json, vectorizes them,
    and uses cosine similarity to find the best matching intent.
    """

    # Confidence thresholds for layered pipeline
    CONFIDENCE_HIGH = 0.65    # Direct intent match
    CONFIDENCE_MEDIUM = 0.40  # Similarity match
    CONFIDENCE_LOW = 0.20     # FAQ retrieval

    def __init__(self, intents_path: Path) -> None:
        self._intents_path = intents_path
        self._intents: list[dict] = []
        self._patterns: list[str] = []
        self._pattern_tags: list[str] = []
        self._tag_to_intent: dict[str, dict] = {}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._is_fitted = False

        self._load_and_fit()

    def _load_and_fit(self) -> None:
        """Load intents and fit the TF-IDF vectorizer."""
        with open(self._intents_path, 'r') as f:
            data = json.load(f)

        self._intents = data['intents']
        self._patterns = []
        self._pattern_tags = []
        self._tag_to_intent = {}

        for intent in self._intents:
            tag = intent['tag']
            self._tag_to_intent[tag] = intent

            for pattern in intent.get('patterns', []):
                preprocessed = preprocess(pattern)
                if preprocessed:
                    self._patterns.append(preprocessed)
                    self._pattern_tags.append(tag)

        if not self._patterns:
            raise ValueError("No patterns found in intents.json")

        # Fit TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=1,
            max_features=5000,
            sublinear_tf=True,       # log normalization
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._patterns)
        self._is_fitted = True

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify the intent of input text.

        Returns a ClassificationResult with intent, confidence, and response.
        """
        start_time = time.time()

        if not text or not text.strip():
            return self._fallback(latency_ms=0.0)

        # Preprocess input
        processed = preprocess(text)
        if not processed:
            processed = text.lower().strip()

        # Vectorize and compute similarity
        try:
            query_vec = self._vectorizer.transform([processed])
            similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        except Exception:
            return self._fallback(latency_ms=(time.time() - start_time) * 1000)

        # Find best matching pattern
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_tag = self._pattern_tags[best_idx]

        latency_ms = (time.time() - start_time) * 1000

        if best_score >= self.CONFIDENCE_HIGH:
            return self._build_result(
                tag=best_tag,
                confidence=best_score,
                source="intent",
                matched_pattern=self._patterns[best_idx],
                latency_ms=latency_ms,
            )

        if best_score >= self.CONFIDENCE_MEDIUM:
            return self._build_result(
                tag=best_tag,
                confidence=best_score,
                source="similarity",
                matched_pattern=self._patterns[best_idx],
                latency_ms=latency_ms,
            )

        return self._fallback(latency_ms=latency_ms)

    def _build_result(
        self,
        tag: str,
        confidence: float,
        source: str,
        matched_pattern: str = "",
        latency_ms: float = 0.0,
    ) -> ClassificationResult:
        """Build a ClassificationResult for a given tag."""
        intent_data = self._tag_to_intent.get(tag, {})
        responses = intent_data.get('responses', [])
        response = random.choice(responses) if responses else self._get_fallback_text()

        return ClassificationResult(
            intent=tag,
            confidence=round(confidence, 3),
            response=response,
            source=source,
            matched_pattern=matched_pattern,
            latency_ms=round(latency_ms, 2),
        )

    def _fallback(self, latency_ms: float = 0.0) -> ClassificationResult:
        """Return a graceful fallback response."""
        fallback_intent = self._tag_to_intent.get('unknown', {})
        responses = fallback_intent.get('responses', [self._get_fallback_text()])
        return ClassificationResult(
            intent="unknown",
            confidence=0.0,
            response=random.choice(responses),
            source="fallback",
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _get_fallback_text() -> str:
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that? I'm best at Python, DSA, Git, Docker, cloud, and career topics.",
            "Hmm, that's outside my current knowledge. Try asking about coding, algorithms, DevOps, or tech careers!",
            "I didn't quite catch that. Could you be more specific? I excel at tech questions — Python, DSA, system design, and more.",
        ]
        return random.choice(fallbacks)

    def get_top_n(self, text: str, n: int = 3) -> list[ClassificationResult]:
        """Get top-N intent matches (useful for ambiguous queries)."""
        processed = preprocess(text) or text.lower()
        try:
            query_vec = self._vectorizer.transform([processed])
            similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        except Exception:
            return [self._fallback()]

        # Get indices of top N
        top_indices = np.argsort(similarities)[::-1][:n]
        seen_tags: set[str] = set()
        results = []

        for idx in top_indices:
            tag = self._pattern_tags[idx]
            score = float(similarities[idx])
            if tag not in seen_tags and score > 0.1:
                seen_tags.add(tag)
                results.append(self._build_result(
                    tag=tag,
                    confidence=score,
                    source="similarity",
                    matched_pattern=self._patterns[idx],
                ))

        return results or [self._fallback()]

    def get_intent_names(self) -> list[str]:
        """Return all known intent tags."""
        return list(self._tag_to_intent.keys())

    def reload(self) -> None:
        """Reload intents from file (hot reload support)."""
        self._load_and_fit()
