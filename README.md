# 🤖 NexusAI — Intelligent NLP Chatbot

A production-ready, portfolio-worthy chatbot built with Python and Streamlit. Features a full NLP pipeline, 5-level hybrid response engine, conversational context memory, and a premium glassmorphism dark-mode UI.

---

## ✨ Features

### NLP Engine
- **Text preprocessing** — normalization, punctuation cleaning, tokenization, stopword removal, lemmatization
- **TF-IDF vectorization** with cosine similarity intent matching
- **Confidence scoring** on every response
- **Synonym expansion** (e.g. `k8s` → `kubernetes`, `dp` → `dynamic programming`)
- **Conservative spell correction** using edit-distance with domain vocabulary
- **Sentiment detection** via VADER — adapts response tone
- **Context-aware pronoun resolution** — "What is Python?" → "Who created **it**?" resolves correctly

### 5-Level Hybrid Response Pipeline
| Level | Trigger | Source |
|-------|---------|--------|
| 1 | Confidence ≥ 0.65 | Direct intent match |
| 2 | Confidence ≥ 0.40 | TF-IDF similarity |
| 3 | FAQ score ≥ 0.45 | Knowledge-base retrieval |
| 4 | Confidence ≥ 0.25 | Clarification prompt |
| 5 | Below all thresholds | Graceful fallback |

### Context Awareness
- Tracks current **topic**, **intent**, and **session history**
- Detects follow-up questions (short queries, pronoun use, same topic)
- Resolves `it / that / this / they` → previous topic noun

### 55+ Intents across domains
Greetings · Python (OOP, decorators, generators, async, lambdas, testing) · DSA (arrays, linked lists, trees, graphs, binary search, sorting, Big-O, DP, recursion) · Git/GitHub · Docker/Kubernetes · Cloud (AWS, Azure, GCP) · Debugging · APIs · Databases · ML basics · CI/CD · Web scraping · Linux · Design patterns · Interview prep · System design · Resume · Career advice · Productivity · Learning resources

### Premium UI
- Dark glassmorphism design with CSS variables
- ChatGPT-style chat layout — user bubbles right, bot bubbles left
- Confidence badge + source badge on every bot message
- Typing indicator animation
- Suggestion cards on empty state
- Sidebar: quick queries, session analytics, top intents, recent chats
- Fixed rounded input bar with Enter-to-send

---

## 🚀 Quick Start

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd chatbot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> NLTK data (punkt, stopwords, wordnet, vader_lexicon) downloads automatically on first run.

---

## 📁 Project Structure

```
chatbot/
├── app.py                        # Streamlit entry point + full UI
├── requirements.txt
├── README.md
│
├── chatbot_engine/               # NLP backend
│   ├── __init__.py
│   ├── preprocess.py             # NLP pipeline (normalize → tokenize → lemmatize)
│   ├── intent_classifier.py      # TF-IDF vectorizer + cosine similarity
│   ├── response_engine.py        # 5-level hybrid pipeline orchestrator
│   ├── context_manager.py        # Conversation memory & pronoun resolution
│   ├── sentiment.py              # VADER sentiment analysis
│   └── spell_corrector.py        # Conservative edit-distance spell correction
│
├── data/
│   ├── intents.json              # 55+ intents with patterns & responses
│   └── faq_knowledge.json        # 8 detailed FAQ entries (Python internals, Git, etc.)
│
└── assets/                       # Static assets (logo, icons)
```

---

## ⚙️ Configuration

### Adding new intents
Edit `data/intents.json`:
```json
{
  "tag": "my_topic",
  "patterns": [
    "question pattern one",
    "another way to ask",
    "variant phrasing"
  ],
  "responses": [
    "Your markdown response here. Supports **bold**, `code`, and \\n\\n```python\\ncode blocks\\n```"
  ],
  "context_set": "my_topic"
}
```
More patterns = better matching. Aim for 5–10 patterns per intent.

### Adding FAQ entries
Edit `data/faq_knowledge.json`:
```json
{
  "question": "What is the difference between X and Y?",
  "answer": "Detailed markdown answer...",
  "tags": ["topic1", "topic2"]
}
```

### Confidence thresholds (response_engine.py)
```python
CONFIDENCE_HIGH   = 0.65   # direct intent match
CONFIDENCE_MEDIUM = 0.40   # similarity match
FAQ_THRESHOLD     = 0.45   # FAQ retrieval
AMBIGUITY_THRESHOLD = 0.25 # ask for clarification
```

---

## 🧩 Architecture

```
User Input
    │
    ▼
Spell Correction ──→ (conservative, domain-aware)
    │
    ▼
Sentiment Analysis ──→ (VADER, adjusts response prefix)
    │
    ▼
Context: is_followup? ──→ Pronoun Resolution ("it" → "Python")
    │
    ▼
TF-IDF Vectorizer ──→ Cosine Similarity
    │
    ├── conf ≥ 0.65 ──→ [L1] Intent Response
    ├── conf ≥ 0.40 ──→ [L2] Similarity Response
    ├── FAQ match   ──→ [L3] Knowledge Base Answer
    ├── conf ≥ 0.25 ──→ [L4] Clarification Prompt
    └── else        ──→ [L5] Graceful Fallback
    │
    ▼
Context Manager: update topic, intent, history
    │
    ▼
EngineResponse → Streamlit UI
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.35 | Web UI |
| nltk | ≥ 3.8 | Tokenization, stopwords, lemmatization, VADER |
| scikit-learn | ≥ 1.4 | TF-IDF, cosine similarity |
| numpy | ≥ 1.26 | Array operations |

---

## 🎓 Extending the Project

**Add spaCy NER:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```
Then integrate in `preprocess.py` using `nlp(text).ents`.

**Add a database backend:**
Replace the in-memory `ContextManager` history with SQLite or Redis for persistent session storage.

**Add voice input:**
Use `streamlit-webrtc` or `SpeechRecognition` to transcribe audio and pass to the engine.

**Deploy to cloud:**
```bash
# Streamlit Cloud — push to GitHub, connect at share.streamlit.io
# Or Docker:
docker build -t nexusai .
docker run -p 8501:8501 nexusai
```

---

## 📄 License

MIT — free to use, modify, and distribute.
