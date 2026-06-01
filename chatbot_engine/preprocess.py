"""
preprocess.py — NLP preprocessing pipeline
Handles: normalization, tokenization, stopword removal, lemmatization, stemming
"""

import re
import string
from typing import Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data on first run
def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_data()

# Singleton instances for performance
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words('english'))

# Keep some question words that carry meaning for chatbot context
_KEEP_WORDS = {'what', 'who', 'when', 'where', 'why', 'how', 'is', 'are',
               'was', 'were', 'do', 'does', 'can', 'could', 'should', 'would'}
_CHATBOT_STOP_WORDS = _stop_words - _KEEP_WORDS

# Synonym map for expanding query coverage
SYNONYM_MAP: dict[str, str] = {
    "ai": "machine learning",
    "artificial intelligence": "machine learning",
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "gh": "github",
    "repo": "repository",
    "fn": "function",
    "func": "function",
    "dict": "dictionary",
    "oop": "object oriented programming",
    "async": "asynchronous",
    "stdlib": "standard library",
    "venv": "virtual environment",
    "pip": "package manager",
    "algo": "algorithm",
    "algos": "algorithms",
    "dp": "dynamic programming",
    "bst": "binary search tree",
    "dfs": "depth first search",
    "bfs": "breadth first search",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "devops": "development operations",
    "vm": "virtual machine",
    "db": "database",
    "sql": "structured query language",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "api": "application programming interface",
    "ui": "user interface",
    "ux": "user experience",
    "os": "operating system",
    "ide": "integrated development environment",
}


def normalize_text(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return ' '.join(text.lower().strip().split())


def clean_punctuation(text: str) -> str:
    """Remove punctuation except apostrophes (for contractions)."""
    # Keep hyphens in compound words, remove the rest
    text = re.sub(r"[^\w\s'-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def expand_synonyms(text: str) -> str:
    """Replace known abbreviations/synonyms with canonical forms."""
    words = text.split()
    expanded = []
    i = 0
    while i < len(words):
        # Try bigrams first
        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in SYNONYM_MAP:
                expanded.append(SYNONYM_MAP[bigram])
                i += 2
                continue
        # Then unigrams
        word = words[i]
        expanded.append(SYNONYM_MAP.get(word, word))
        i += 1
    return ' '.join(expanded)


def tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords while preserving question words."""
    return [t for t in tokens if t not in _CHATBOT_STOP_WORDS and len(t) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Apply WordNet lemmatization."""
    return [_lemmatizer.lemmatize(token) for token in tokens]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Apply Porter stemming (optional, more aggressive than lemmatization)."""
    return [_stemmer.stem(token) for token in tokens]


def preprocess(
    text: str,
    use_stemming: bool = False,
    remove_stops: bool = True,
    expand_syns: bool = True,
) -> str:
    """
    Full NLP preprocessing pipeline.

    Args:
        text: Raw input text
        use_stemming: Apply stemming after lemmatization
        remove_stops: Remove stopwords
        expand_syns: Expand abbreviations/synonyms

    Returns:
        Preprocessed string ready for vectorization
    """
    if not text or not text.strip():
        return ""

    # Step 1: Normalize
    text = normalize_text(text)

    # Step 2: Expand synonyms
    if expand_syns:
        text = expand_synonyms(text)

    # Step 3: Clean punctuation
    text = clean_punctuation(text)

    # Step 4: Tokenize
    tokens = tokenize(text)

    # Step 5: Remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # Step 6: Lemmatize
    tokens = lemmatize_tokens(tokens)

    # Step 7: Optional stemming
    if use_stemming:
        tokens = stem_tokens(tokens)

    return ' '.join(tokens)


def preprocess_for_display(text: str) -> str:
    """
    Lighter preprocessing for display/logging — preserves more structure.
    """
    text = normalize_text(text)
    text = clean_punctuation(text)
    return text


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract the most meaningful keywords from text."""
    processed = preprocess(text, remove_stops=True)
    tokens = processed.split()
    # Filter very short tokens
    keywords = [t for t in tokens if len(t) > 2]
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:top_n]
