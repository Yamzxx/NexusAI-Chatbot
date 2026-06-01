"""
response_engine.py — 5-level hybrid response pipeline
Level 1: High-confidence intent matching  (>= 0.65)
Level 2: Medium-confidence TF-IDF        (>= 0.40)
Level 3: FAQ knowledge-base retrieval    (>= 0.45)
Level 4: Clarification prompt            (>= 0.25)
Level 5: Graceful fallback               (catch-all)
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

from .intent_classifier import IntentClassifier, ClassificationResult
from .context_manager import ContextManager
from .sentiment import SentimentAnalyzer, SentimentResult
from .spell_corrector import SpellCorrector
from .preprocess import preprocess


@dataclass
class EngineResponse:
    """Full response payload from the engine."""
    text: str
    intent: str
    confidence: float
    source: str           # intent | similarity | faq | clarify | fallback
    latency_ms: float
    sentiment: Optional[SentimentResult] = None
    was_spell_corrected: bool = False
    corrected_input: str = ""
    is_followup: bool = False
    topic: str = ""


class ResponseEngine:
    """
    Hybrid NLP engine — never fails silently.
    Context-aware with pronoun resolution and conversation memory.
    """

    AMBIGUITY_THRESHOLD = 0.25

    CLARIFICATION_PROMPTS = [
        "Could you be more specific? I want to give you the most helpful answer possible.",
        "I'm not entirely sure what you mean — could you rephrase or add more context?",
        "That's a bit ambiguous. Are you asking about a specific language, tool, or concept?",
        "Can you clarify? I'm great at Python, DSA, Git, Docker, cloud topics, interviews, and career advice.",
    ]

    def __init__(self, intents_path: Path, faq_path: Path) -> None:
        self._classifier  = IntentClassifier(intents_path)
        self._context     = ContextManager()
        self._sentiment   = SentimentAnalyzer()
        self._spell       = SpellCorrector()

        self._faq_items: list[dict] = []
        self._faq_vectorizer: Optional[TfidfVectorizer] = None
        self._faq_matrix  = None
        self._load_faq(faq_path)

        # Analytics
        self._total: int = 0
        self._failed: int = 0
        self._intent_counts: dict[str, int] = {}
        self._conf_sum: float = 0.0

    # ── FAQ ────────────────────────────────────────────────────────────────

    def _load_faq(self, path: Path) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            self._faq_items = data.get("faq", [])
        except Exception:
            return
        if not self._faq_items:
            return
        qs = [preprocess(item["question"]) for item in self._faq_items]
        self._faq_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._faq_matrix = self._faq_vectorizer.fit_transform(qs)

    def _search_faq(self, text: str) -> Optional[tuple[str, float]]:
        if not self._faq_items or self._faq_vectorizer is None:
            return None
        proc = preprocess(text) or text.lower()
        try:
            vec  = self._faq_vectorizer.transform([proc])
            sims = cosine_similarity(vec, self._faq_matrix).flatten()
            idx  = int(np.argmax(sims))
            score = float(sims[idx])
            if score >= 0.45:
                return self._faq_items[idx]["answer"], score
        except Exception:
            pass
        return None

    # ── Sentiment ──────────────────────────────────────────────────────────

    def _prefix(self, response: str, sentiment: SentimentResult) -> str:
        pfx = self._sentiment.get_response_prefix(sentiment)
        if pfx and not response.startswith(("I understand", "No worries", "Glad")):
            return pfx + response
        return response

    # ── Main generate ──────────────────────────────────────────────────────

    def generate(self, raw_input: str) -> EngineResponse:
        t0 = time.time()
        self._total += 1

        if not raw_input or not raw_input.strip():
            return self._resp("Please type your question!", "empty", 0.0, "validation", t0)

        # 1. Spell-correct
        corrected, was_fixed = self._spell.correct_sentence(raw_input)
        working = corrected

        # 2. Sentiment (on original)
        sentiment = self._sentiment.analyze(raw_input)

        # 3. Context: detect follow-up BEFORE resolving pronouns
        is_followup = self._context.is_followup_question(working)

        # 4. Pronoun resolution (inject topic noun into query)
        if is_followup:
            working = self._context.resolve_reference(working)

        # ── LEVEL 1 & 2: Intent classifier ────────────────────────────────
        clf = self._classifier.classify(working)

        if clf.confidence >= self._classifier.CONFIDENCE_MEDIUM:
            text = self._prefix(clf.response, sentiment)
            self._track(clf.intent, clf.confidence)
            self._context.update(raw_input, text, clf.intent, clf.confidence)
            return self._resp(text, clf.intent, clf.confidence, clf.source, t0,
                              sentiment, was_fixed,
                              corrected if was_fixed else "",
                              is_followup,
                              self._context.state.current_topic)

        # ── LEVEL 3: FAQ ──────────────────────────────────────────────────
        faq = self._search_faq(working)
        if faq:
            answer, score = faq
            text = self._prefix(answer, sentiment)
            self._track("faq_retrieval", score)
            self._context.update(raw_input, text, "faq_retrieval", score)
            return self._resp(text, "faq_retrieval", score, "faq", t0,
                              sentiment, was_fixed,
                              corrected if was_fixed else "",
                              is_followup,
                              self._context.state.current_topic)

        # ── LEVEL 4: Clarification ────────────────────────────────────────
        if clf.confidence >= self.AMBIGUITY_THRESHOLD:
            top2 = self._classifier.get_top_n(working, n=2)
            clarify = random.choice(self.CLARIFICATION_PROMPTS)
            if len(top2) >= 2 and top2[0].confidence > 0.2:
                a = top2[0].intent.replace("_", " ").title()
                b = top2[1].intent.replace("_", " ").title()
                clarify = (f"I found a couple of possible topics — are you asking about "
                           f"**{a}** or **{b}**? A bit more context will help me nail it!")
            self._track("clarification", clf.confidence)
            self._context.update(raw_input, clarify, "clarification", clf.confidence)
            return self._resp(clarify, "clarification", clf.confidence, "clarify", t0,
                              sentiment, was_fixed, corrected if was_fixed else "", is_followup)

        # ── LEVEL 5: Fallback ─────────────────────────────────────────────
        self._failed += 1
        text = clf.response
        self._track("unknown", 0.0)
        self._context.update(raw_input, text, "unknown", 0.0)
        return self._resp(text, "unknown", 0.0, "fallback", t0,
                          sentiment, was_fixed, corrected if was_fixed else "", is_followup)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _resp(
        self, text: str, intent: str, conf: float, source: str, t0: float,
        sentiment: Optional[SentimentResult] = None,
        was_fixed: bool = False, corrected: str = "",
        is_followup: bool = False, topic: str = "",
    ) -> EngineResponse:
        return EngineResponse(
            text=text, intent=intent, confidence=conf, source=source,
            latency_ms=round((time.time() - t0) * 1000, 2),
            sentiment=sentiment,
            was_spell_corrected=was_fixed, corrected_input=corrected,
            is_followup=is_followup, topic=topic,
        )

    def _track(self, intent: str, conf: float) -> None:
        self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1
        self._conf_sum += conf

    # ── Analytics & session ────────────────────────────────────────────────

    def get_analytics(self) -> dict:
        avg = (self._conf_sum / self._total) if self._total else 0.0
        top = sorted(self._intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "total_queries":   self._total,
            "failed_queries":  self._failed,
            "success_rate":    round((1 - self._failed / self._total) * 100, 1) if self._total else 100.0,
            "avg_confidence":  round(avg * 100, 1),
            "top_intents":     top,
            "session_info":    self._context.get_conversation_summary(),
        }

    def reset_session(self) -> None:
        self._context.reset()
        self._total = self._failed = 0
        self._intent_counts.clear()
        self._conf_sum = 0.0

    @property
    def context(self) -> ContextManager:
        return self._context
