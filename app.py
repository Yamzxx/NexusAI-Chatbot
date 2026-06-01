"""
app.py — NexusAI Chatbot  |  streamlit run app.py
Premium glassmorphism dark-mode UI with full NLP intelligence.
"""

import streamlit as st
import time
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

from chatbot_engine import ResponseEngine, EngineResponse

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusAI — Intelligent Tech Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #07090f;
  --surface:    #0d1117;
  --glass:      rgba(255,255,255,0.035);
  --glass-h:    rgba(255,255,255,0.06);
  --border:     rgba(255,255,255,0.07);
  --border-a:   rgba(96,165,250,0.28);
  --cyan:       #38bdf8;
  --blue:       #6366f1;
  --violet:     #a78bfa;
  --green:      #34d399;
  --red:        #f87171;
  --orange:     #fb923c;
  --t1:         #eef2f7;
  --t2:         #94a3b8;
  --t3:         #475569;
  --mono:       'JetBrains Mono', monospace;
  --font:       'Outfit', sans-serif;
  --r-xl:       18px;
  --r-lg:       13px;
  --r-md:       9px;
  --r-sm:       5px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
  font-family: var(--font) !important;
  color: var(--t1) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
.stDeployButton { display: none !important; }

.main .block-container {
  padding: 0 2rem 9rem !important;
  max-width: 920px !important;
  margin: 0 auto !important;
}

/* ─── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.4rem 1.1rem !important; }

.sb-logo {
  display: flex; align-items: center; gap: 11px;
  padding: .65rem .9rem;
  background: var(--glass); border: 1px solid var(--border);
  border-radius: var(--r-lg); margin-bottom: 1.4rem;
}
.sb-logo-icon { font-size: 1.9rem; line-height: 1; }
.sb-logo h2 {
  font-size: .95rem; font-weight: 700; letter-spacing: -.01em;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-logo p { font-size: .66rem; color: var(--t3); margin-top: 1px; }

.sb-title {
  font-size: .62rem; font-weight: 700; color: var(--t3);
  text-transform: uppercase; letter-spacing: .1em;
  margin: 1.2rem 0 .55rem; padding-left: 2px;
}
.sb-divider { height: 1px; background: var(--border); margin: 1rem 0; }

/* sidebar buttons */
div[data-testid="stButton"] > button {
  width: 100% !important; text-align: left !important;
  background: var(--glass) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; color: var(--t2) !important;
  font-family: var(--font) !important; font-size: .76rem !important;
  padding: .42rem .8rem !important; margin-bottom: .3rem !important;
  transition: all .18s !important;
}
div[data-testid="stButton"] > button:hover {
  background: var(--glass-h) !important; border-color: var(--border-a) !important;
  color: var(--t1) !important; transform: translateX(3px) !important;
  box-shadow: 0 0 14px rgba(56,189,248,.08) !important;
}

/* analytics metric cards */
.mc {
  background: var(--glass); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: .55rem .8rem;
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: .35rem;
}
.mc-label { font-size: .68rem; color: var(--t3); }
.mc-val   { font-size: .85rem; font-weight: 700; color: var(--cyan); font-family: var(--mono); }

.intent-row {
  display: flex; justify-content: space-between;
  padding: 3px 0; font-size: .7rem; border-bottom: 1px solid var(--border);
  color: var(--t2);
}
.intent-row:last-child { border: none; }
.ic { color: var(--cyan); font-family: var(--mono); font-weight: 600; }

.recent-item {
  padding: .42rem .7rem; background: var(--glass); border: 1px solid var(--border);
  border-radius: var(--r-sm); margin-bottom: .3rem;
  font-size: .69rem; color: var(--t2);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ─── Header ───────────────────────────────────────────────────────────── */
.chat-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: .9rem 1.4rem; background: var(--glass);
  border: 1px solid var(--border); border-radius: var(--r-xl);
  margin: 1.4rem 0 1.5rem; backdrop-filter: blur(16px);
  box-shadow: 0 0 40px rgba(99,179,237,.07); position: relative; overflow: hidden;
}
.chat-header::before {
  content:''; position: absolute; top:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg,transparent,rgba(99,179,237,.35),transparent);
}
.ch-left  { display: flex; align-items: center; gap: 13px; }
.ch-avatar {
  width:44px; height:44px; border-radius:11px;
  background: linear-gradient(135deg, var(--blue), var(--violet));
  display:flex; align-items:center; justify-content:center; font-size:1.4rem;
  box-shadow: 0 4px 14px rgba(99,102,241,.4);
}
.ch-info h1 { font-size:1.05rem; font-weight:700; letter-spacing:-.02em;
  background: linear-gradient(135deg,var(--cyan),var(--violet));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.ch-info p  { font-size:.68rem; color:var(--t3); margin-top:2px; }
.status-dot {
  display:inline-block; width:6px; height:6px; background:var(--green);
  border-radius:50%; margin-right:5px; box-shadow: 0 0 5px var(--green);
  animation: pulse-g 2s infinite;
}
@keyframes pulse-g {
  0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.4)}
}
.ch-right { display:flex; gap:7px; }
.hbadge {
  font-size:.6rem; font-weight:600; padding:3px 9px; border-radius:20px;
  font-family:var(--mono); letter-spacing:.04em;
}
.hb-c { background:rgba(56,189,248,.1); color:var(--cyan); border:1px solid rgba(56,189,248,.2); }
.hb-v { background:rgba(167,139,250,.1); color:var(--violet); border:1px solid rgba(167,139,250,.2); }

/* ─── Hero ─────────────────────────────────────────────────────────────── */
.hero {
  text-align:center; padding: 3rem 1rem 2rem;
  animation: fadeUp .5s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
.hero-icon { font-size:3.5rem; display:inline-block; animation:float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
.hero h2 {
  font-size:1.75rem; font-weight:800; letter-spacing:-.03em; margin:.8rem 0 .5rem;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { font-size:.88rem; color:var(--t2); line-height:1.65; max-width:440px; margin:0 auto 2.5rem; }

/* ─── Messages ─────────────────────────────────────────────────────────── */
.msg-row { display:flex; gap:11px; margin-bottom:1.2rem; animation: msgIn .3s ease; }
@keyframes msgIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
.msg-row.user { flex-direction:row-reverse; }

.av {
  width:34px; height:34px; border-radius:9px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-size:.95rem;
  margin-top:2px;
}
.av-bot  { background: linear-gradient(135deg,var(--blue),var(--violet)); }
.av-user { background: linear-gradient(135deg,#1d4ed8,#1e40af); }

.msg-wrap { max-width:80%; }
.msg-row.user .msg-wrap { align-items:flex-end; display:flex; flex-direction:column; }

.bubble {
  padding:.8rem 1.05rem; border-radius:var(--r-lg); font-size:.865rem;
  line-height:1.7; word-wrap:break-word;
}
.bubble-user {
  background: linear-gradient(135deg,#1d4ed8,#4338ca);
  color:#fff; border-bottom-right-radius:var(--r-sm);
  box-shadow: 0 4px 14px rgba(29,78,216,.3);
}
.bubble-bot {
  background: var(--glass); border:1px solid var(--border); color:var(--t1);
  border-bottom-left-radius:var(--r-sm); box-shadow:0 6px 24px rgba(0,0,0,.35);
}

/* code inside bot bubbles */
.bubble-bot pre {
  background: rgba(0,0,0,.5) !important; border:1px solid rgba(255,255,255,.07) !important;
  border-radius:var(--r-sm) !important; padding:.7rem !important;
  overflow-x:auto; margin:.5rem 0 !important; font-family:var(--mono) !important;
  font-size:.775rem !important;
}
.bubble-bot code { font-family:var(--mono) !important; font-size:.8rem !important;
  background:rgba(0,0,0,.35) !important; padding:2px 5px !important;
  border-radius:3px !important; }

.msg-meta {
  display:flex; gap:7px; align-items:center; flex-wrap:wrap; margin-top:4px;
}
.msg-row.user .msg-meta { justify-content:flex-end; }
.ts { font-size:.59rem; color:var(--t3); font-family:var(--mono); }

/* badges */
.bdg {
  font-size:.57rem; font-weight:700; padding:2px 7px; border-radius:20px;
  font-family:var(--mono); letter-spacing:.03em; text-transform:uppercase;
}
.bdg-intent    {background:rgba(52,211,153,.1); color:var(--green);  border:1px solid rgba(52,211,153,.22);}
.bdg-similarity{background:rgba(56,189,248,.1); color:var(--cyan);   border:1px solid rgba(56,189,248,.22);}
.bdg-faq       {background:rgba(251,146,60,.1); color:var(--orange); border:1px solid rgba(251,146,60,.22);}
.bdg-clarify   {background:rgba(167,139,250,.1);color:var(--violet); border:1px solid rgba(167,139,250,.22);}
.bdg-fallback  {background:rgba(248,113,113,.1);color:var(--red);    border:1px solid rgba(248,113,113,.22);}
.bdg-ch        {background:rgba(52,211,153,.1); color:var(--green);  border:1px solid rgba(52,211,153,.22);}
.bdg-cm        {background:rgba(251,146,60,.1); color:var(--orange); border:1px solid rgba(251,146,60,.22);}
.bdg-cl        {background:rgba(248,113,113,.1);color:var(--red);    border:1px solid rgba(248,113,113,.22);}
.bdg-fu        {background:rgba(99,102,241,.1); color:#818cf8;       border:1px solid rgba(99,102,241,.22);}

.spell-note { font-size:.63rem; color:var(--t3); font-style:italic; margin-bottom:3px; }

/* ─── Typing indicator ─────────────────────────────────────────────────── */
.typing-row { display:flex; gap:11px; align-items:center; margin-bottom:1.2rem; }
.typing-bubble {
  display:flex; align-items:center; gap:9px;
  padding:.65rem .95rem; background:var(--glass); border:1px solid var(--border);
  border-radius:var(--r-lg); border-bottom-left-radius:var(--r-sm);
}
.t-dots { display:flex; gap:5px; }
.t-dot {
  width:7px; height:7px; border-radius:50%;
  animation: tb 1.2s infinite ease-in-out;
}
.t-dot:nth-child(1){background:var(--cyan);}
.t-dot:nth-child(2){background:var(--violet);animation-delay:.2s;}
.t-dot:nth-child(3){background:var(--blue);animation-delay:.4s;}
@keyframes tb {
  0%,60%,100%{transform:translateY(0);opacity:.5}
  30%{transform:translateY(-5px);opacity:1}
}
.t-label { font-size:.7rem; color:var(--t3); font-style:italic; }

/* ─── Input bar ────────────────────────────────────────────────────────── */
.input-bar {
  position:fixed; bottom:0; left:0; right:0; z-index:200;
  padding:.9rem 1.5rem 1.1rem;
  background: linear-gradient(to top, var(--bg) 75%, transparent);
}
.input-inner {
  max-width:920px; margin:0 auto;
  background:var(--glass); border:1px solid var(--border);
  border-radius:22px; padding:.6rem .7rem;
  backdrop-filter:blur(20px);
  box-shadow: 0 -2px 30px rgba(0,0,0,.3), 0 0 0 0 rgba(56,189,248,0);
  transition: border-color .2s, box-shadow .2s;
}
.input-inner:focus-within {
  border-color: var(--border-a);
  box-shadow: 0 -2px 30px rgba(0,0,0,.3), 0 0 18px rgba(56,189,248,.1);
}

/* override Streamlit textarea inside input bar */
.input-inner textarea {
  background: transparent !important; border: none !important;
  box-shadow: none !important; color: var(--t1) !important;
  font-family: var(--font) !important; font-size: .88rem !important;
  resize: none !important;
}
.input-inner textarea::placeholder { color: var(--t3) !important; }
.input-inner [data-testid="stTextArea"] > div { border:none !important; padding:0 !important; }
.input-inner div[data-testid="stButton"] > button {
  background: linear-gradient(135deg,var(--blue),var(--violet)) !important;
  border: none !important; border-radius: 10px !important;
  color: #fff !important; font-size: 1.15rem !important;
  width: 44px !important; height: 44px !important; min-height: 44px !important;
  padding: 0 !important; transform: none !important;
  box-shadow: 0 3px 12px rgba(99,102,241,.45) !important;
  margin: 0 !important;
}
.input-inner div[data-testid="stButton"] > button:hover {
  transform: scale(1.07) !important;
  box-shadow: 0 5px 18px rgba(99,102,241,.65) !important;
  border-color: transparent !important;
}

/* clear button */
.clr div[data-testid="stButton"] > button {
  background: rgba(248,113,113,.07) !important;
  border: 1px solid rgba(248,113,113,.18) !important;
  color: var(--red) !important; font-size: .73rem !important;
  padding: .4rem .8rem !important; margin-top: .4rem !important;
}
.clr div[data-testid="stButton"] > button:hover {
  background: rgba(248,113,113,.14) !important;
  border-color: var(--red) !important;
}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.12)}
</style>
"""

# ─── Engine ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine() -> ResponseEngine:
    return ResponseEngine(
        intents_path=DATA_DIR / "intents.json",
        faq_path=DATA_DIR / "faq_knowledge.json",
    )


# ─── Session state ────────────────────────────────────────────────────────────
def init_state() -> None:
    for k, v in {
        "messages": [],
        "input_key": 0,
        "pending": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.now().strftime("%H:%M")

def conf_class(s: float) -> str:
    return "bdg-ch" if s >= .65 else "bdg-cm" if s >= .40 else "bdg-cl"

SOURCE_LABELS = {
    "intent":     ("bdg-intent",     "✦ Intent"),
    "similarity": ("bdg-similarity", "◈ Similar"),
    "faq":        ("bdg-faq",        "◉ Knowledge Base"),
    "clarify":    ("bdg-clarify",    "⊙ Clarifying"),
    "fallback":   ("bdg-fallback",   "⊘ Fallback"),
}

def conf_dot(s: float) -> str:
    return "●" if s >= .75 else "◑" if s >= .50 else "◔" if s >= .25 else "○"


def render_msg(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]
    stamp = msg.get("ts", "")
    m = msg.get("meta", {})

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
          <div class="av av-user">👤</div>
          <div class="msg-wrap">
            <div class="bubble bubble-user">{content}</div>
            <div class="msg-meta"><span class="ts">{stamp}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        conf   = m.get("confidence", 0.0)
        src    = m.get("source", "fallback")
        lat    = m.get("latency_ms", 0.0)
        fixed  = m.get("was_spell_corrected", False)
        cinput = m.get("corrected_input", "")
        fu     = m.get("is_followup", False)
        semoji = m.get("sentiment_emoji", "")

        src_cls, src_lbl = SOURCE_LABELS.get(src, ("bdg-fallback", src.title()))
        cc = conf_class(conf)
        pct = f"{int(conf * 100)}%"

        spell_html = (f'<div class="spell-note">✱ Auto-corrected: "{cinput}"</div>' if fixed and cinput else "")
        fu_html    = ('<span class="bdg bdg-fu">↩ follow-up</span>' if fu else "")
        sem_html   = (f'<span style="font-size:.75rem">{semoji}</span>' if semoji else "")

        st.markdown(f"""
        <div class="msg-row">
          <div class="av av-bot">🤖</div>
          <div class="msg-wrap">
            {spell_html}
            <div class="bubble bubble-bot">
        """, unsafe_allow_html=True)

        st.markdown(content)  # renders markdown, code blocks, tables

        st.markdown(f"""
            </div>
            <div class="msg-meta">
              <span class="ts">{stamp} · {lat:.0f}ms</span>
              <span class="bdg {src_cls}">{src_lbl}</span>
              <span class="bdg {cc}">{conf_dot(conf)} {pct}</span>
              {fu_html}{sem_html}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


def render_typing() -> None:
    st.markdown("""
    <div class="typing-row">
      <div class="av av-bot">🤖</div>
      <div class="typing-bubble">
        <div class="t-dots">
          <div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div>
        </div>
        <span class="t-label">NexusAI is thinking…</span>
      </div>
    </div>""", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
    <div class="hero">
      <div class="hero-icon">🤖</div>
      <h2>Welcome to NexusAI</h2>
      <p>Your intelligent tech assistant — powered by NLP & TF-IDF.<br>
         Ask me anything about Python, DSA, Git, Docker, cloud, interviews, or career advice.</p>
    </div>""", unsafe_allow_html=True)

    suggestions = [
        ("🐍", "What is Python and what is it used for?"),
        ("🔍", "Explain binary search with code"),
        ("🌿", "What are the most important Git commands?"),
        ("🐳", "How does Docker work and why use it?"),
        ("🎯", "How should I prepare for coding interviews?"),
        ("☁️", "What are the core AWS services?"),
        ("⚡", "What are Python generators and decorators?"),
        ("🚀", "How do I get my first developer job?"),
    ]
    cols = st.columns(4)
    for i, (icon, text) in enumerate(suggestions):
        with cols[i % 4]:
            label = f"{icon} {text[:32]}{'…' if len(text)>32 else ''}"
            if st.button(label, key=f"sg_{i}", use_container_width=True):
                st.session_state.pending = text
                st.rerun()


def process(engine: ResponseEngine, raw: str) -> None:
    raw = raw.strip()
    if not raw:
        return

    st.session_state.messages.append({"role": "user", "content": raw, "ts": ts()})

    resp: EngineResponse = engine.generate(raw)

    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.text,
        "ts": ts(),
        "meta": {
            "confidence":         resp.confidence,
            "source":             resp.source,
            "latency_ms":         resp.latency_ms,
            "was_spell_corrected": resp.was_spell_corrected,
            "corrected_input":    resp.corrected_input,
            "is_followup":        resp.is_followup,
            "sentiment_emoji":    resp.sentiment.emoji if resp.sentiment else "",
        },
    })


def render_sidebar(engine: ResponseEngine) -> None:
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
          <div class="sb-logo-icon">🤖</div>
          <div>
            <h2>NexusAI</h2>
            <p>Intelligent Tech Assistant</p>
          </div>
        </div>""", unsafe_allow_html=True)

        # Quick queries
        st.markdown('<div class="sb-title">⚡ Quick Queries</div>', unsafe_allow_html=True)
        quick = [
            ("🐍", "Python OOP & Classes"),
            ("🌳", "Binary Trees explained"),
            ("⚡", "Python async/await"),
            ("λ",  "Python lambda functions"),
            ("🔧", "Git branching strategy"),
            ("🐳", "Docker Compose basics"),
            ("☁️", "AWS core services"),
            ("📊", "Big-O notation guide"),
            ("🔄", "Dynamic programming"),
            ("🧪", "Python unit testing"),
            ("🐛", "Debugging strategies"),
            ("💼", "Resume tips for devs"),
        ]
        for icon, lbl in quick:
            if st.button(f"{icon}  {lbl}", key=f"q_{lbl}"):
                st.session_state.pending = lbl
                st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Analytics
        a = engine.get_analytics()
        si = a.get("session_info", {})
        st.markdown('<div class="sb-title">📊 Analytics</div>', unsafe_allow_html=True)
        for lbl, val in [
            ("Total Queries",   a["total_queries"]),
            ("Avg Confidence",  f"{a['avg_confidence']}%"),
            ("Success Rate",    f"{a['success_rate']}%"),
            ("Failed Queries",  a["failed_queries"]),
            ("Turns",           si.get("turn_count", 0)),
            ("Topic",           (si.get("current_topic") or "—").title()),
        ]:
            st.markdown(
                f'<div class="mc"><span class="mc-label">{lbl}</span>'
                f'<span class="mc-val">{val}</span></div>',
                unsafe_allow_html=True,
            )

        top = a.get("top_intents", [])
        if top:
            st.markdown('<div class="sb-title" style="margin-top:.9rem">🎯 Top Intents</div>', unsafe_allow_html=True)
            for intent, cnt in top[:5]:
                disp = intent.replace("_", " ").title()
                st.markdown(
                    f'<div class="intent-row"><span>{disp}</span>'
                    f'<span class="ic">{cnt}</span></div>',
                    unsafe_allow_html=True,
                )

        # Recent chats
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"][-4:]
        if user_msgs:
            st.markdown('<div class="sb-title" style="margin-top:.9rem">🕐 Recent</div>', unsafe_allow_html=True)
            for msg in reversed(user_msgs):
                preview = msg["content"][:44] + ("…" if len(msg["content"]) > 44 else "")
                st.markdown(f'<div class="recent-item">💬 {preview}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Clear
        st.markdown('<div class="clr">', unsafe_allow_html=True)
        if st.button("🗑️  Clear Conversation", key="clr_btn", use_container_width=True):
            st.session_state.messages = []
            engine.reset_session()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="text-align:center;margin-top:1.4rem;font-size:.6rem;color:var(--t3)">'
            'NexusAI v2.0 · NLP-Powered<br>Python + Streamlit + TF-IDF</div>',
            unsafe_allow_html=True,
        )


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    with st.spinner("Initialising NexusAI…"):
        engine = load_engine()

    render_sidebar(engine)

    # Header
    st.markdown("""
    <div class="chat-header">
      <div class="ch-left">
        <div class="ch-avatar">🤖</div>
        <div class="ch-info">
          <h1>NexusAI</h1>
          <p><span class="status-dot"></span>Online · Intelligent Tech Assistant</p>
        </div>
      </div>
      <div class="ch-right">
        <span class="hbadge hb-c">NLP · TF-IDF</span>
        <span class="hbadge hb-v">55 Intents</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Handle pending (from suggestion cards / sidebar buttons)
    if st.session_state.pending:
        q = st.session_state.pending
        st.session_state.pending = None
        process(engine, q)

    # Chat history
    if not st.session_state.messages:
        render_hero()
    else:
        for msg in st.session_state.messages:
            render_msg(msg)

    # Input form (handles Enter key via form submit)
    st.markdown('<div class="input-bar"><div class="input-inner">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([12, 1])
        with c1:
            user_input = st.text_area(
                label="",
                placeholder="Ask anything — Python, DSA, Git, Docker, cloud, interviews…",
                height=52,
                label_visibility="collapsed",
                key=f"inp_{st.session_state.input_key}",
            )
        with c2:
            submitted = st.form_submit_button("➤", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if submitted and user_input and user_input.strip():
        st.session_state.input_key += 1
        process(engine, user_input)
        st.rerun()


if __name__ == "__main__":
    main()
