# 🤖 NexusAI — Intelligent NLP Chatbot

A chatbot built using Python and Streamlit.

This project was created to learn how chatbots work using Natural Language Processing (NLP). It can understand different user questions, match them with suitable responses, and remember the conversation context for follow-up questions.
---

## What it can do

- Understand user queries using NLP
 1.Match questions using TF-IDF and cosine similarity
 2.Handle follow-up questions using conversation context
 3.Detect sentiment using VADER
 4.Correct small spelling mistakes
 5.Answer questions from a knowledge base
 6.Show confidence scores for responses
 7.Provide fallback responses when it cannot understand a query

## Technologies Used
- Python
- Streamlit
- NLTK
- Scikit-Learn
- NumPy


## How It Works

- When a user sends a message:

 1.The text is cleaned and processed.
 2.Spelling mistakes are checked.
 3.Sentiment is detected.
 4.The chatbot looks for the best matching intent.
 5.If no strong match is found, it checks the FAQ knowledge base.
 6.The response is returned and the conversation context is updated.

### 55+ Intents across domains
Greetings · Python (OOP, decorators, generators, async, lambdas, testing) · DSA (arrays, linked lists, trees, graphs, binary search, sorting, Big-O, DP, recursion) · Git/GitHub · Docker/Kubernetes · Cloud (AWS, Azure, GCP) · Debugging · APIs · Databases · ML basics · CI/CD · Web scraping · Linux · Design patterns · Interview prep · System design · Resume · Career advice · Productivity · Learning resources



---

##  Running the Project

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

## Project Structure

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


##  Architecture

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



## Future Improvements

Voice input support
Database storage
More intents and responses
Better context tracking
Integration with external APIs

## Why I Built This

I wanted to learn more about NLP and chatbot development beyond simple keyword-based bots. This project helped me understand text preprocessing, intent classification, similarity matching, sentiment analysis, and conversation management.

## Thanks for checking out the project!

⭐ Feel free to star the repository and suggest any improvements.

## 📄 License

MIT — free to use, modify, and distribute.
