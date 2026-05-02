"""
Streamlit frontend for the Ecommerce AI Assistant.
Connects to the RAG + Agent pipeline from Backend/Rag/main_05.py
"""

import sys
import os
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_frontend_dir = os.path.dirname(os.path.abspath(__file__))
_rag_dir      = os.path.join(_frontend_dir, "..", "Backend", "Rag")
_agent_dir    = os.path.join(_frontend_dir, "..", "Backend", "Agent")

sys.path.insert(0, _rag_dir)
sys.path.insert(0, _agent_dir)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_rag_dir, ".env"))
load_dotenv(dotenv_path=os.path.join(_agent_dir, "db", ".env"))

# ── Backend imports ───────────────────────────────────────────────────────────
from main_05 import (
    detect_intent,
    run_db_agent,
    rag_answer,
    collection_exists,
    ingest,
)
from retriver_03 import get_vectorstore
from rag_chain_04 import get_llm
from tools.database_tools import check_stock, buy_product

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ecommerce AI Assistant",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Full dark theme CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Force entire app dark ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    section[data-testid="stSidebar"],
    .main, .block-container {
        background-color: #0d0d0d !important;
        color: #e2e8f0 !important;
    }

    /* Hide the Streamlit top bar / hamburger / deploy button */
    [data-testid="stHeader"]          { display: none !important; }
    [data-testid="stToolbar"]         { display: none !important; }
    [data-testid="stDecoration"]      { display: none !important; }
    #MainMenu                         { display: none !important; }
    footer                            { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }

    /* ── Block container padding ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 780px !important;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 28px 0 6px 0;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Status pills row ── */
    .pills-row {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin: 16px 0 24px 0;
        flex-wrap: wrap;
    }
    .pill {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 0.4px;
    }
    .pill-purple { background: #1e1030; color: #a78bfa; border: 1px solid #4c1d95; }
    .pill-blue   { background: #0c1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }
    .pill-green  { background: #0c2218; color: #34d399; border: 1px solid #065f46; }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #1e1e2e;
        margin: 0 0 20px 0;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: #fff;
        padding: 13px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 6px 0 6px auto;
        max-width: 72%;
        font-size: 14.5px;
        line-height: 1.6;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.25);
        word-wrap: break-word;
    }

    .bot-bubble {
        background: #161625;
        color: #cbd5e1;
        padding: 13px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 6px auto 6px 0;
        max-width: 72%;
        font-size: 14.5px;
        line-height: 1.6;
        border: 1px solid #252540;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        word-wrap: break-word;
    }

    /* ── Intent badges ── */
    .badge {
        font-size: 10px;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 8px;
        margin-bottom: 7px;
        display: inline-block;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .badge-rag   { background: #0c2218; color: #34d399; border: 1px solid #065f46; }
    .badge-buy   { background: #2d0f0f; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-stock { background: #0c1a2e; color: #60a5fa; border: 1px solid #1e3a5f; }

    /* ── Welcome card ── */
    .welcome-card {
        background: linear-gradient(135deg, #13132a, #0d1f35);
        border: 1px solid #252550;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 10px 0 20px 0;
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
    }
    .welcome-card strong {
        color: #a78bfa;
    }

    /* ── Suggestion chips ── */
    .chips-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        justify-content: center;
        margin: 14px 0 0 0;
    }
    .chip {
        background: #1a1a30;
        color: #94a3b8;
        border: 1px solid #2d2d50;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px;
        cursor: default;
    }

    /* ── Input box ── */
    .stTextInput > div > div > input {
        background-color: #161625 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d2d50 !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        caret-color: #a78bfa !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6d28d9 !important;
        box-shadow: 0 0 0 2px rgba(109,40,217,0.2) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }

    /* ── Send button ── */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px !important;
        transition: opacity 0.2s !important;
    }
    .stFormSubmitButton > button:hover {
        opacity: 0.85 !important;
    }

    /* ── Clear button ── */
    .stButton > button {
        background: #1a1a2e !important;
        color: #64748b !important;
        border: 1px solid #252540 !important;
        border-radius: 10px !important;
        font-size: 12px !important;
        padding: 4px 14px !important;
    }
    .stButton > button:hover {
        border-color: #f87171 !important;
        color: #f87171 !important;
    }

    /* ── Spinner text ── */
    .stSpinner > div { color: #a78bfa !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #252540; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline init (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    if not collection_exists():
        ingest()
    vectorstore    = get_vectorstore()
    llm            = get_llm()
    db_tools       = [check_stock, buy_product]
    tools_by_name  = {t.name: t for t in db_tools}
    llm_with_tools = llm.bind_tools(db_tools)
    return vectorstore, llm, llm_with_tools, tools_by_name


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🛍️ Ecommerce AI Assistant</h1>
    <p>Ask about products · Check stock · Place orders</p>
</div>
<div class="pills-row">
    <span class="pill pill-purple">🤖 LLaMA 3.3 70B</span>
    <span class="pill pill-blue">🗄️ Qdrant RAG</span>
    <span class="pill pill-green">🛒 Supabase</span>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ── Load pipeline ─────────────────────────────────────────────────────────────
with st.spinner("Loading AI pipeline..."):
    try:
        vectorstore, llm, llm_with_tools, tools_by_name = init_pipeline()
    except Exception as e:
        st.error(f"❌ Failed to initialise pipeline: {e}")
        st.stop()


# ── Intent badge map ──────────────────────────────────────────────────────────
BADGE = {
    "rag":   '<span class="badge badge-rag">📚 Product Info</span>',
    "stock": '<span class="badge badge-stock">📦 Stock Check</span>',
    "buy":   '<span class="badge badge-buy">🛒 Purchase</span>',
}


# ── Welcome card (shown when no messages) ─────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div class="welcome-card">
    � Hi! I'm your <strong>AI shopping assistant</strong>.<br>
    I can answer product questions, check live stock, and process purchases.<br><br>
    <div class="chips-row">
        <span class="chip">💬 "What is Product 5?"</span>
        <span class="chip">📦 "Is Product 12 in stock?"</span>
        <span class="chip">🛒 "Buy Product 8"</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑&nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        badge = BADGE.get(msg.get("intent", "rag"), "")
        st.markdown(
            f'<div class="bot-bubble">{badge}<br>🤖&nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True,
        )


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="msg",
            placeholder="Ask about a product, check stock, or buy something...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

# Clear chat button (small, below input)
col_a, col_b, col_c = st.columns([3, 2, 3])
with col_b:
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Handle submission ─────────────────────────────────────────────────────────
if submitted and user_input.strip():
    query  = user_input.strip()
    intent = detect_intent(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        try:
            if intent in ("buy", "stock"):
                answer = run_db_agent(llm_with_tools, tools_by_name, query)
            else:
                answer = rag_answer(query, vectorstore, llm)
        except Exception as e:
            answer = f"❌ Something went wrong: {str(e)}"
            intent = "rag"

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": intent,
    })

    st.rerun()
