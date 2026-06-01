"""
spell_corrector.py — Conservative spell correction
Only corrects clearly misspelled words, never changes valid words.
"""
import re
from typing import Optional


class SpellCorrector:
    """
    Conservative spell corrector. Uses a large tech/English vocabulary
    and only corrects words with edit-distance-1 that clearly map to
    a known word. Does NOT correct words already in vocabulary.
    """

    # Full domain vocabulary — everything valid should be here
    DOMAIN_VOCAB: set[str] = {
        # Common English
        "what", "who", "when", "where", "why", "how", "which", "is", "are",
        "was", "were", "do", "does", "did", "can", "could", "should", "would",
        "will", "have", "has", "had", "be", "been", "being", "a", "an", "the",
        "in", "on", "at", "to", "for", "of", "and", "or", "but", "not", "no",
        "yes", "it", "its", "this", "that", "these", "those", "they", "their",
        "there", "here", "with", "from", "by", "about", "into", "through",
        "then", "than", "so", "if", "as", "up", "down", "out", "off", "over",
        "also", "just", "more", "some", "any", "all", "very", "too", "my",
        "your", "his", "her", "our", "we", "me", "him", "us", "you", "i",
        "tell", "explain", "show", "give", "make", "use", "create", "write",
        "build", "run", "find", "get", "set", "add", "remove", "delete",
        "read", "load", "save", "open", "close", "start", "stop", "new",
        "old", "good", "bad", "best", "first", "last", "next", "same", "other",
        "difference", "between", "define", "definition", "example", "examples",
        "implement", "implementation", "understand", "learn", "know", "need",
        "want", "like", "work", "working", "works", "help", "used", "using",
        "created", "create", "creator", "made", "make", "makes", "designed",
        "developed", "invented", "built", "written", "released", "introduced",
        "called", "called", "named", "means", "mean", "called", "called",
        "hello", "hi", "hey", "bye", "goodbye", "thanks", "thank", "please",
        "sorry", "okay", "ok", "sure", "great", "nice", "cool", "awesome",
        # Programming
        "python", "javascript", "typescript", "java", "golang", "go", "rust",
        "cpp", "csharp", "ruby", "php", "swift", "kotlin", "scala", "haskell",
        "code", "coding", "program", "programming", "script", "scripting",
        "function", "functions", "method", "methods", "class", "classes",
        "object", "objects", "variable", "variables", "constant", "constants",
        "parameter", "parameters", "argument", "arguments", "return", "returns",
        "loop", "loops", "iteration", "iterate", "recursive", "recursion",
        "condition", "conditions", "conditional", "boolean", "string", "integer",
        "float", "list", "dictionary", "tuple", "set", "array", "arrays",
        "module", "modules", "package", "packages", "library", "libraries",
        "import", "imports", "export", "exports", "install", "installed",
        "decorator", "decorators", "generator", "generators", "iterator",
        "lambda", "closure", "closures", "memoization", "caching", "cache",
        "async", "await", "asynchronous", "synchronous", "concurrent",
        "thread", "threads", "threading", "process", "processes", "multiprocessing",
        "exception", "exceptions", "error", "errors", "debug", "debugging",
        "test", "testing", "unittest", "pytest", "assertion", "mock",
        "type", "types", "typing", "annotation", "annotations", "hints",
        "oop", "inheritance", "polymorphism", "encapsulation", "abstraction",
        "interface", "abstract", "singleton", "factory", "observer", "pattern",
        # DSA
        "algorithm", "algorithms", "complexity", "space", "time",
        "sorting", "sort", "search", "searching", "binary", "linear",
        "quicksort", "mergesort", "heapsort", "bubblesort", "insertionsort",
        "tree", "trees", "graph", "graphs", "node", "nodes", "edge", "edges",
        "stack", "queue", "heap", "linked", "traversal", "breadth", "depth",
        "dynamic", "greedy", "backtracking", "divide", "conquer",
        "hash", "hashing", "collision", "bucket", "probing",
        "big", "notation", "logarithm", "polynomial", "exponential",
        # Git
        "git", "github", "gitlab", "bitbucket", "commit", "commits",
        "branch", "branches", "merge", "merging", "rebase", "rebasing",
        "clone", "fork", "pull", "push", "remote", "origin", "upstream",
        "stash", "cherry", "pick", "tag", "release", "workflow",
        "repository", "repo", "conflict", "resolve", "resolution",
        # Docker / DevOps
        "docker", "container", "containers", "image", "images", "volume",
        "kubernetes", "k8s", "pod", "pods", "deployment", "service",
        "dockerfile", "compose", "registry", "orchestration",
        "ci", "cd", "pipeline", "jenkins", "github", "actions", "devops",
        "terraform", "ansible", "infrastructure", "microservices", "serverless",
        # Cloud
        "aws", "azure", "gcp", "cloud", "ec2", "s3", "lambda", "rds",
        "vpc", "iam", "cloudfront", "sagemaker", "bedrock", "region",
        "availability", "zone", "bucket", "blob", "function", "compute",
        "storage", "networking", "security", "firewall", "gateway",
        # ML/Data
        "machine", "learning", "deep", "neural", "network", "networks",
        "model", "models", "training", "dataset", "features", "labels",
        "supervised", "unsupervised", "reinforcement", "classification",
        "regression", "clustering", "embedding", "transformer", "attention",
        "tensorflow", "pytorch", "sklearn", "scikit", "pandas", "numpy",
        "matplotlib", "seaborn", "visualization", "dataframe", "series",
        # Career
        "interview", "resume", "cv", "career", "job", "salary", "negotiate",
        "leetcode", "faang", "google", "amazon", "meta", "microsoft", "apple",
        "linkedin", "portfolio", "project", "projects", "experience",
        "productivity", "pomodoro", "focus", "habit", "routine",
    }

    def __init__(self) -> None:
        self.vocab = self.DOMAIN_VOCAB.copy()

    def correct_word(self, word: str) -> str:
        """
        Return corrected word ONLY if it's clearly wrong and has an obvious fix.
        Conservative: if unsure, return original.
        """
        w = word.lower().strip()
        # Always accept: short words, words in vocab, words with digits/special chars
        if len(w) <= 3 or w in self.vocab or re.search(r'[\d@/\-_]', w):
            return w

        # Try edit-distance-1 candidates
        candidates = self._edits1(w) & self.vocab
        if len(candidates) == 1:
            return candidates.pop()  # Unambiguous single correction
        # If multiple candidates, don't guess — return original
        return w

    def _edits1(self, word: str) -> set[str]:
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:]           for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces   = [L + c + R[1:]       for L, R in splits if R for c in letters]
        inserts    = [L + c + R           for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def correct_sentence(self, sentence: str) -> tuple[str, bool]:
        """
        Correct sentence. Returns (corrected, was_changed).
        Only corrects words longer than 4 chars with a clear single fix.
        """
        if not sentence or not sentence.strip():
            return sentence, False
        words = sentence.lower().split()
        corrected = []
        changed = False
        for word in words:
            fixed = self.correct_word(word)
            if fixed != word and len(word) > 4:
                changed = True
            else:
                fixed = word  # keep original case/value if no confident fix
            corrected.append(fixed)
        return ' '.join(corrected), changed

    def add_words(self, words: list[str]) -> None:
        self.vocab.update(w.lower() for w in words)
