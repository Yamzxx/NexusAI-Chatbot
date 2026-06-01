"""
context_manager.py — Conversational context and session memory management
Tracks previous topics, entities, intents, and resolves pronouns/references
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import re
import time


@dataclass
class Turn:
    """A single conversation turn."""
    user_input: str
    bot_response: str
    intent: str
    topic: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    entities: dict = field(default_factory=dict)


@dataclass
class ContextState:
    """Current conversation context snapshot."""
    current_topic: str = ""
    current_intent: str = ""
    last_entity: str = ""
    turn_count: int = 0
    is_followup: bool = False


class ContextManager:
    """
    Manages lightweight conversation memory for context-aware responses.

    Handles:
    - Topic tracking across turns
    - Pronoun resolution ("it", "that", "this", "they")
    - Follow-up detection
    - Session history
    """

    # Pronoun patterns that indicate reference to previous topic
    REFERENCE_PATTERNS = [
        r'\bit\b', r'\bthat\b', r'\bthis\b', r'\bthey\b',
        r'\btheir\b', r'\bthose\b', r'\bsame\b',
    ]

    # Topic keyword map for quick topic detection
    TOPIC_KEYWORDS: dict[str, list[str]] = {
        "python": ["python", "pip", "venv", "django", "flask", "fastapi", "pandas",
                   "numpy", "async", "decorator", "generator", "lambda", "class",
                   "function", "module", "package", "import"],
        "dsa": ["array", "list", "tree", "graph", "stack", "queue", "heap",
                "algorithm", "sorting", "searching", "dynamic programming",
                "recursion", "complexity", "big o", "hash", "linked"],
        "git": ["git", "github", "gitlab", "commit", "branch", "merge", "rebase",
                "clone", "fork", "pull request", "pr", "remote", "origin"],
        "docker": ["docker", "container", "image", "kubernetes", "k8s", "pod",
                   "deploy", "dockerfile", "compose", "registry", "volume"],
        "cloud": ["aws", "azure", "gcp", "cloud", "lambda", "ec2", "s3",
                  "serverless", "microservices", "terraform"],
        "interview": ["interview", "leetcode", "faang", "system design", "resume",
                      "career", "job", "salary", "negotiation"],
        "ml": ["machine learning", "deep learning", "neural", "model", "training",
               "dataset", "tensorflow", "pytorch", "scikit", "pandas"],
        "debugging": ["bug", "error", "debug", "fix", "traceback", "exception",
                      "crash", "fail", "broken", "issue"],
    }

    def __init__(self, max_history: int = 10) -> None:
        self._history: deque[Turn] = deque(maxlen=max_history)
        self._context = ContextState()
        self._session_start = time.time()

    @property
    def state(self) -> ContextState:
        return self._context

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._context.turn_count

    def detect_topic(self, text: str) -> str:
        """Detect the primary topic from text using keyword matching."""
        text_lower = text.lower()
        scores: dict[str, int] = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[topic] = score
        return max(scores, key=scores.get) if scores else ""

    def has_pronoun_reference(self, text: str) -> bool:
        """Check if the input contains a pronoun/reference word."""
        text_lower = text.lower()
        return any(re.search(pat, text_lower) for pat in self.REFERENCE_PATTERNS)

    def resolve_reference(self, text: str) -> str:
        """
        Resolve pronoun references by injecting previous topic context.

        Example:
            Previous topic: "Python"
            Input: "Who created it?"
            Output: "Who created Python?"
        """
        if not self._context.current_topic or not self.has_pronoun_reference(text):
            return text

        topic = self._context.current_topic
        topic_display = topic.title()
        text_lower = text.lower()

        # Replace "it" and similar with the topic name
        for pattern in [r'\bit\b', r'\bthat\b', r'\bthis\b']:
            if re.search(pattern, text_lower):
                resolved = re.sub(pattern, topic_display, text, flags=re.IGNORECASE)
                return resolved

        return text

    def is_followup_question(self, text: str) -> bool:
        """
        Determine if this is a follow-up to the previous turn.
        """
        if not self._history:
            return False

        # Very short question (< 5 words) is likely a follow-up
        words = text.split()
        if len(words) <= 5 and self._context.current_topic:
            return True

        # Contains pronoun reference
        if self.has_pronoun_reference(text):
            return True

        # Starts with a conjunction suggesting continuation
        followup_starters = ['and', 'but', 'also', 'what about', 'how about',
                              'then', 'so', 'any', 'more about', 'tell me more',
                              'can you explain', 'could you', 'what else', 'anything else']
        text_lower = text.lower().strip()
        if any(text_lower.startswith(s) for s in followup_starters):
            return True

        # Same topic as previous turn
        new_topic = self.detect_topic(text)
        if new_topic and new_topic == self._context.current_topic:
            return True

        return False

    def get_context_hint(self) -> str:
        """
        Return a context hint to inject into response generation.
        """
        if not self._history:
            return ""

        parts = []
        if self._context.current_topic:
            parts.append(f"Current topic: {self._context.current_topic}")
        if self._context.current_intent:
            parts.append(f"Previous intent: {self._context.current_intent}")
        if self._context.is_followup:
            parts.append("This is a follow-up question")

        return " | ".join(parts)

    def update(
        self,
        user_input: str,
        bot_response: str,
        intent: str,
        confidence: float,
        entities: Optional[dict] = None,
    ) -> None:
        """
        Update context after each conversation turn.
        """
        # Detect topic from current input
        new_topic = self.detect_topic(user_input)
        if new_topic:
            self._context.current_topic = new_topic

        self._context.current_intent = intent
        self._context.turn_count += 1
        self._context.is_followup = self.is_followup_question(user_input)

        turn = Turn(
            user_input=user_input,
            bot_response=bot_response[:200],  # Store truncated for memory efficiency
            intent=intent,
            topic=new_topic or self._context.current_topic,
            confidence=confidence,
            entities=entities or {},
        )
        self._history.append(turn)

    def get_recent_intents(self, n: int = 5) -> list[str]:
        """Get the n most recent intents."""
        turns = list(self._history)[-n:]
        return [t.intent for t in turns]

    def get_session_duration(self) -> float:
        """Return session duration in seconds."""
        return time.time() - self._session_start

    def reset(self) -> None:
        """Reset context for a new session."""
        self._history.clear()
        self._context = ContextState()
        self._session_start = time.time()

    def get_conversation_summary(self) -> dict:
        """Return a summary of the conversation for analytics."""
        return {
            "turn_count": self._context.turn_count,
            "current_topic": self._context.current_topic,
            "unique_topics": list({t.topic for t in self._history if t.topic}),
            "avg_confidence": (
                sum(t.confidence for t in self._history) / len(self._history)
                if self._history else 0.0
            ),
            "session_duration_s": round(self.get_session_duration(), 1),
        }
