#  NexusAI Chatbot

A chatbot built using Python and Streamlit.

This project was created to learn how chatbots work using Natural Language Processing (NLP). It can understand different user questions, match them with suitable responses, and remember conversation context for follow-up questions.

---

##  What It Can Do

* Understand user queries using NLP
* Match questions using TF-IDF and cosine similarity
* Handle follow-up questions using conversation context
* Detect sentiment using VADER
* Correct small spelling mistakes
* Answer questions from a knowledge base
* Show confidence scores for responses
* Provide fallback responses when it cannot understand a query

---

##  Technologies Used

* Python
* Streamlit
* NLTK
* Scikit-Learn
* NumPy

---

##  How It Works

When a user sends a message:

1. The text is cleaned and processed.
2. Spelling mistakes are checked.
3. Sentiment is detected.
4. The chatbot looks for the best matching intent.
5. If no strong match is found, it checks the FAQ knowledge base.
6. The response is returned and the conversation context is updated.

---

##  Supported Topics

The chatbot currently supports 55+ intents across different domains:

* Greetings
* Python (OOP, decorators, generators, async, lambdas, testing)
* Data Structures & Algorithms
* Git & GitHub
* Docker & Kubernetes
* Cloud Computing (AWS, Azure, GCP)
* APIs
* Databases
* Machine Learning Basics
* CI/CD
* Linux
* Web Scraping
* Design Patterns
* System Design
* Interview Preparation
* Resume Building
* Career Guidance
* Productivity Tips
* Learning Resources

---

##  Running the Project

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

in your browser.

> NLTK resources (punkt, stopwords, wordnet, vader_lexicon) will be downloaded automatically during the first run.

---

## 📂 Project Structure

```text
chatbot/
├── app.py
├── requirements.txt
├── README.md
│
├── chatbot_engine/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── intent_classifier.py
│   ├── response_engine.py
│   ├── context_manager.py
│   ├── sentiment.py
│   └── spell_corrector.py
│
├── data/
│   ├── intents.json
│   └── faq_knowledge.json
│
└── assets/
```

---

## 🏗️ Architecture

```text
User Input
    │
    ▼
Text Processing
    │
    ▼
Spell Correction
    │
    ▼
Sentiment Analysis
    │
    ▼
Context Handling
    │
    ▼
TF-IDF + Similarity Matching
    │
    ├── Intent Match
    ├── FAQ Match
    ├── Clarification
    └── Fallback
    │
    ▼
Response Generation
    │
    ▼
Streamlit UI
```

---

## 🔮 Future Improvements

* Voice input support
* Database storage
* More intents and responses
* Better context tracking
* Integration with external APIs
* Support for additional NLP models

---

## 💡 Why I Built This

I wanted to learn more about NLP and chatbot development beyond simple keyword-based bots. This project helped me understand text preprocessing, intent classification, similarity matching, sentiment analysis, and conversation management.

---

## ⭐ Support

Thanks for checking out the project!

If you found it useful, feel free to give the repository a star.

Suggestions, improvements, and contributions are always welcome.

---

## 📄 License

This project is licensed under the MIT License.
