"""
sentiment.py — Sentiment detection and emotional tone analysis
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK
"""

from dataclasses import dataclass
from typing import Literal
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def _ensure_vader() -> None:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)


_ensure_vader()


@dataclass
class SentimentResult:
    label: Literal["positive", "neutral", "negative"]
    score: float          # -1.0 to 1.0
    compound: float       # raw VADER compound
    positive: float
    negative: float
    neutral: float
    emoji: str
    tone_hint: str        # response guidance hint


class SentimentAnalyzer:
    """Lightweight sentiment analyzer using VADER."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of input text.

        Returns SentimentResult with label, scores, and response hints.
        """
        if not text or not text.strip():
            return SentimentResult(
                label="neutral", score=0.0, compound=0.0,
                positive=0.0, negative=0.0, neutral=1.0,
                emoji="😐", tone_hint="standard"
            )

        scores = self._analyzer.polarity_scores(text)
        compound = scores['compound']

        # Determine label
        if compound >= 0.05:
            label: Literal["positive", "neutral", "negative"] = "positive"
            emoji = "😊"
            tone_hint = "enthusiastic"
        elif compound <= -0.05:
            label = "negative"
            emoji = "😟"
            tone_hint = "empathetic"
        else:
            label = "neutral"
            emoji = "😐"
            tone_hint = "standard"

        # Detect frustration signals
        frustration_words = {'frustrated', 'confused', 'stuck', 'lost', 'wrong',
                              'broken', 'error', 'doesnt work', "doesn't work",
                              'not working', 'help me', 'please help', 'annoying'}
        text_lower = text.lower()
        if any(word in text_lower for word in frustration_words):
            tone_hint = "supportive"
            if label != "negative":
                label = "negative"
                emoji = "😤"

        # Detect excitement
        excitement_signals = text.count('!') >= 2 or any(
            w in text_lower for w in ['amazing', 'awesome', 'love', 'great', 'perfect']
        )
        if excitement_signals and compound > 0:
            tone_hint = "celebratory"
            emoji = "🎉"

        return SentimentResult(
            label=label,
            score=round(compound, 3),
            compound=compound,
            positive=round(scores['pos'], 3),
            negative=round(scores['neg'], 3),
            neutral=round(scores['neu'], 3),
            emoji=emoji,
            tone_hint=tone_hint,
        )

    def get_response_prefix(self, result: SentimentResult) -> str:
        """Generate a brief empathetic prefix based on sentiment."""
        prefixes = {
            "enthusiastic": ["Great question! ", "Love the enthusiasm! ", ""],
            "celebratory": ["Awesome! 🎉 ", "That's great! ", ""],
            "supportive": ["I understand your frustration. Let me help. ",
                           "No worries, let's work through this together. "],
            "empathetic": ["I understand. ", "Let me help with that. ", ""],
            "standard": ["", "", ""],  # Usually no prefix for neutral
        }
        import random
        options = prefixes.get(result.tone_hint, [""])
        return random.choice(options)
