"""
app.py – CounsellAI
Main Streamlit application with a full LangGraph agentic pipeline.

Architecture:
  User Input
      │
      ▼
  [Router Node]  ──► JSON: {category, needs_abroad}
      │
      ├─► [Retriever Node]        ← metadata-filtered Chroma query
      │         │
      │         ├─► [AptitudeQuiz Node]   (career queries only)
      │         ├─► [EaseScore Node]      (abroad queries only)
      │         └─► [ResponseGen Node]   ← main counselor LLM
      │                   │
      └──────────────────► [Documentation Node]  ← session PDF
                                │
                            Streamlit UI

Run:  streamlit run app.py
"""

import os
import json
import re
import datetime
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated
import operator

import streamlit as st

# LangGraph
from langgraph.graph import StateGraph, END

# LangChain
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# LLM – tries OpenAI first, falls back to Groq, then raises helpful error
try:
    from langchain_openai import ChatOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# PDF generation
from fpdf import FPDF

# Internal
from prompts import (
    COUNSELOR_SYSTEM_PROMPT,
    ROUTER_PROMPT,
    EASE_SCORE_PROMPT,
    APTITUDE_QUIZ_PROMPT,
    APTITUDE_SCORE_PROMPT,
    SESSION_SUMMARY_PROMPT,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "counsellai"
SESSION_DIR     = Path("./session_records")
SESSION_DIR.mkdir(exist_ok=True)

TOP_K = 6   # Number of chunks to retrieve


# ══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_llm():
    """Return the best available LLM."""
    openai_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    groq_key   = os.getenv("GROQ_API_KEY",   st.secrets.get("GROQ_API_KEY",   ""))

    if _OPENAI_AVAILABLE and openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_key,
            streaming=True,
        )
    elif _GROQ_AVAILABLE and groq_key:
        return ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            api_key=groq_key,
            streaming=True,
        )
    else:
        st.error(
            "⚠️ No LLM API key found.\n\n"
            "Set **OPENAI_API_KEY** or **GROQ_API_KEY** as an environment variable "
            "or in `.streamlit/secrets.toml`."
        )
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Optional[Chroma]:
    if not Path(CHROMA_DIR).exists():
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    query:           str
    profile:         str
    category:        str
    needs_abroad:    bool
    countries:       List[str]
    retrieved_docs:  List[dict]          # [{page_content, metadata}]
    context_str:     str
    quiz_questions:  Optional[List[dict]]
    quiz_answers:    Optional[dict]
    ease_score_text: str
    final_response:  str
    sources:         List[str]
    messages:        Annotated[List, operator.add]  # full chat history


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> AgentState:
    """Classify the query → category + needs_abroad flag."""
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip possible markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(raw)
        category     = parsed.get("category", "mixed")
        needs_abroad = bool(parsed.get("needs_abroad", False))
    except Exception:
        category     = "mixed"
        needs_abroad = False

    # Detect country mentions for ease score
    country_map = {
        "usa": "USA", "us": "USA", "united states": "USA",
        "uk": "UK", "united kingdom": "UK",
        "canada": "Canada",
        "australia": "Australia",
        "germany": "Germany",
        "singapore": "Singapore",
        "new zealand": "New Zealand",
    }
    q_lower = state["query"].lower()
    countries = [v for k, v in country_map.items() if k in q_lower]
    countries = list(dict.fromkeys(countries))  # deduplicate, preserve order

    return {**state, "category": category, "needs_abroad": needs_abroad, "countries": countries}


def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from Chroma with metadata filtering."""
    vs = get_vectorstore()
    if vs is None:
        return {**state, "retrieved_docs": [], "context_str": "No data indexed yet. Please run ingest.py first.", "sources": []}

    query       = state["query"]
    category    = state["category"]
    needs_abroad = state["needs_abroad"]

    # Build metadata filter
    # If abroad query, don't restrict by country so we get global + specific results
    if category != "mixed":
        if needs_abroad:
            where_filter = {"category": {"$in": [category, "admissions"]}}
        else:
            where_filter = {"$and": [
                {"category": {"$eq": category}},
                {"country":  {"$in": ["India", "Global"]}},
            ]}
    else:
        where_filter = None

    try:
        if where_filter:
            docs = vs.similarity_search(query, k=TOP_K, filter=where_filter)
        else:
            docs = vs.similarity_search(query, k=TOP_K)
    except Exception:
        # Fallback: no filter
        docs = vs.similarity_search(query, k=TOP_K)

    # Build context string
    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        src      = doc.metadata.get("filename", "unknown")
        page     = doc.metadata.get("page", "")
        page_str = f", p.{page}" if page != "" else ""
        context_parts.append(f"[{i}] [Source: {src}{page_str}]\n{doc.page_content}")
        src_label = f"{src}{page_str}"
        if src_label not in sources:
            sources.append(src_label)

    context_str  = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    raw_docs     = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

    return {**state, "retrieved_docs": raw_docs, "context_str": context_str, "sources": sources}


def aptitude_quiz_node(state: AgentState) -> AgentState:
    """Generate aptitude quiz questions (career category only)."""
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=APTITUDE_QUIZ_PROMPT)])
    raw = response.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
    except Exception:
        questions = []

    # Store questions in session state for UI rendering
    st.session_state["pending_quiz"] = questions
    return {**state, "quiz_questions": questions}


def ease_score_node(state: AgentState) -> AgentState:
    """Calculate Visa + Admission Ease Score for abroad queries."""
    llm     = get_llm()
    countries_str = ", ".join(state["countries"]) if state["countries"] else "General (abroad)"

    prompt = EASE_SCORE_PROMPT.format(
        context=state["context_str"],
        profile=state["profile"],
        countries=countries_str,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "ease_score_text": response.content.strip()}


def response_generator_node(state: AgentState) -> AgentState:
    """Main counselor response using the full structured prompt."""
    llm = get_llm()

    # Prepend ease score info to context if available
    context = state["context_str"]
    if state.get("ease_score_text"):
        context = f"=== VISA + ADMISSION EASE SCORE ===\n{state['ease_score_text']}\n\n=== RETRIEVED DOCUMENTS ===\n{context}"

    # Prepend aptitude insights if quiz was answered
    if state.get("quiz_answers") and state.get("context_str"):
        apt_prompt = APTITUDE_SCORE_PROMPT.format(
            answers=json.dumps(state["quiz_answers"], indent=2),
            context=state["context_str"],
        )
        apt_response = llm.invoke([HumanMessage(content=apt_prompt)])
        context = f"=== APTITUDE ASSESSMENT RESULTS ===\n{apt_response.content}\n\n{context}"

    system_message = COUNSELOR_SYSTEM_PROMPT.format(
        profile=state["profile"],
        context=context,
        question=state["query"],
    )

    messages = [SystemMessage(content=system_message)]
    # Add prior conversation turns for context continuity
    if state.get("messages"):
        messages += state["messages"][-6:]   # last 3 exchanges

    response = llm.invoke(messages)
    final    = response.content.strip()

    return {**state, "final_response": final,
            "messages": [HumanMessage(content=state["query"]), AIMessage(content=final)]}


def documentation_node(state: AgentState) -> AgentState:
    """Generate and store a session PDF record."""
    now         = datetime.datetime.now()
    followup    = (now + datetime.timedelta(days=7)).strftime("%d %B %Y")
    sources_str = "\n".join(f"  • {s}" for s in state.get("sources", []))

    # Build plain-text summary via LLM
    llm = get_llm()
    summary_prompt = SESSION_SUMMARY_PROMPT.format(
        datetime=now.strftime("%d %B %Y, %I:%M %p"),
        profile=state["profile"],
        category=state["category"],
        followup=followup,
        sources=sources_str if sources_str else "None",
        conversation=(
            f"Student: {state['query']}\n\nCounsellor: {state['final_response']}"
        ),
    )
    summary_resp = llm.invoke([HumanMessage(content=summary_prompt)])
    summary_text = summary_resp.content.strip()

    # Build PDF
    pdf_path = _create_session_pdf(summary_text, now)
    st.session_state["last_pdf_path"] = str(pdf_path)

    return state


def _create_session_pdf(summary_text: str, now: datetime.datetime) -> Path:
    """Write a clean one-page PDF session record and return its path."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(37, 99, 235)   # brand blue
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(0, 14, "CounsellAI – Session Record", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 22)
    pdf.cell(0, 6, f"Generated: {now.strftime('%d %B %Y  %I:%M %p')}", ln=True)

    # Body
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 36)
    pdf.set_font("Helvetica", "", 10)
    # Encode to latin-1 safely (FPDF default)
    clean_text = summary_text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 5, clean_text)

    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0, 10,
        "CounsellAI is AI-assisted. For serious personal concerns, consult a licensed professional.",
        align="C",
    )

    filename = f"session_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = SESSION_DIR / filename
    pdf.output(str(pdf_path))
    return pdf_path


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════════

def should_run_quiz(state: AgentState) -> str:
    if state["category"] == "career" and not st.session_state.get("quiz_done"):
        return "aptitude_quiz"
    return "retriever"


def should_run_ease_score(state: AgentState) -> str:
    return "ease_score" if state["needs_abroad"] else "response_generator"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",             router_node)
    g.add_node("aptitude_quiz",      aptitude_quiz_node)
    g.add_node("retriever",          retriever_node)
    g.add_node("ease_score",         ease_score_node)
    g.add_node("response_generator", response_generator_node)
    g.add_node("documentation",      documentation_node)

    g.set_entry_point("router")

    # Router → quiz or retriever
    g.add_conditional_edges("router", should_run_quiz, {
        "aptitude_quiz": "aptitude_quiz",
        "retriever":     "retriever",
    })

    # Quiz always goes to retriever next
    g.add_edge("aptitude_quiz", "retriever")

    # Retriever → ease_score or response_generator
    g.add_conditional_edges("retriever", should_run_ease_score, {
        "ease_score":         "ease_score",
        "response_generator": "response_generator",
    })

    g.add_edge("ease_score",         "response_generator")
    g.add_edge("response_generator", "documentation")
    g.add_edge("documentation",      END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages":    [],
        "profile":     "",
        "filter_mode": "Both",
        "pending_quiz": None,
        "quiz_done":    False,
        "quiz_answers": {},
        "last_pdf_path": None,
        "graph_messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/student-center.png", width=80)
        st.title("CounsellAI")
        st.caption("Your AI Education Counsellor 🎓")
        st.divider()

        st.subheader("🧑‍🎓 Your Profile")
        name    = st.text_input("Name",        placeholder="e.g. Priya Sharma")
        cls     = st.selectbox("Class / Level", ["Class 10", "Class 11", "Class 12",
                                                   "Undergrad (1st yr)", "Undergrad (2nd yr)",
                                                   "Undergrad (3rd yr)", "Undergrad (Final yr)",
                                                   "Post-Graduate"])
        stream  = st.selectbox("Stream / Major", ["Science (PCM)", "Science (PCB)", "Commerce",
                                                    "Arts / Humanities", "Engineering", "BBA/MBA",
                                                    "Medicine / MBBS", "Law", "Other"])
        pct     = st.slider("Current %  / CGPA", 0, 100, 75, help="Enter your latest academic score")
        budget  = st.selectbox("Budget (Abroad study)", ["< ₹10 LPA", "₹10–20 LPA",
                                                           "₹20–40 LPA", "> ₹40 LPA", "Not Applicable"])
        ielts   = st.text_input("IELTS / TOEFL score", placeholder="e.g. 7.0 / 100")
        goals   = st.text_area("Career Goals (optional)", height=60,
                               placeholder="e.g. Become a data scientist…")

        if st.button("💾 Save Profile", use_container_width=True, type="primary"):
            profile_str = (
                f"Name: {name or 'Not provided'} | Level: {cls} | Stream: {stream} | "
                f"Score: {pct}% | Budget: {budget} | IELTS/TOEFL: {ielts or 'Not provided'} | "
                f"Goals: {goals or 'Not specified'}"
            )
            st.session_state["profile"] = profile_str
            st.success("Profile saved!")

        st.divider()
        st.subheader("🔍 Study Scope")
        st.session_state["filter_mode"] = st.radio(
            "Show colleges / universities from:",
            ["India Only", "Abroad Only", "Both"],
            index=2,
        )

        st.divider()
        st.subheader("📁 Upload Marksheet")
        uploaded = st.file_uploader("Upload marksheet (PDF)", type=["pdf"])
        if uploaded:
            st.info(f"📄 {uploaded.name} uploaded. Mention your scores in chat for personalised advice.")

        st.divider()
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state["messages"]       = []
            st.session_state["graph_messages"] = []
            st.session_state["quiz_done"]      = False
            st.session_state["quiz_answers"]   = {}
            st.session_state["pending_quiz"]   = None
            st.session_state["last_pdf_path"]  = None
            st.rerun()

        # Data status
        st.divider()
        db_ok = Path(CHROMA_DIR).exists()
        if db_ok:
            st.success("✅ Knowledge base loaded")
        else:
            st.warning("⚠️ No knowledge base found.\nRun `python ingest.py` first.")


def render_quiz_widget(questions: List[dict]) -> Optional[dict]:
    """Render aptitude quiz inside a Streamlit form; return answers or None."""
    st.info("🧩 **Career Aptitude Quiz** – Answer these 5 quick questions to get personalised career guidance!")
    answers = {}
    with st.form("aptitude_quiz_form"):
        for q in questions:
            st.markdown(f"**Q{q['id']}. {q['question']}**")
            opts = q.get("options", {})
            choice = st.radio(
                f"Select answer for Q{q['id']}",
                options=list(opts.keys()),
                format_func=lambda k, o=opts: f"{k}) {o[k]}",
                key=f"quiz_q_{q['id']}",
                label_visibility="collapsed",
            )
            answers[q["id"]] = {"question": q["question"], "chosen": choice,
                                 "answer_text": opts.get(choice, "")}
            st.markdown("---")
        submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True, type="primary")

    return answers if submitted else None


def run_agent(query: str, profile: str, quiz_answers: Optional[dict] = None) -> dict:
    """Invoke the LangGraph agent and return the final state."""
    graph = build_graph()
    initial_state: AgentState = {
        "query":           query,
        "profile":         profile or "Not provided",
        "category":        "",
        "needs_abroad":    False,
        "countries":       [],
        "retrieved_docs":  [],
        "context_str":     "",
        "quiz_questions":  None,
        "quiz_answers":    quiz_answers or st.session_state.get("quiz_answers"),
        "ease_score_text": "",
        "final_response":  "",
        "sources":         [],
        "messages":        st.session_state.get("graph_messages", []),
    }

    config = {"recursion_limit": 20}
    result = graph.invoke(initial_state, config=config)
    # Save messages for continuity
    st.session_state["graph_messages"] = result.get("messages", [])
    return result


def render_chat():
    st.title("🎓 CounsellAI")
    st.caption("Your AI-powered Education Counsellor for Indian Students")

    # Check knowledge base
    if not Path(CHROMA_DIR).exists():
        st.warning(
            "📚 **Knowledge base not found.**\n\n"
            "Add your PDF/CSV datasets to the `./data/` folder and run:\n"
            "```bash\npython ingest.py\n```"
        )

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending aptitude quiz
    if st.session_state.get("pending_quiz") and not st.session_state.get("quiz_done"):
        with st.chat_message("assistant"):
            answers = render_quiz_widget(st.session_state["pending_quiz"])
            if answers:
                st.session_state["quiz_answers"] = answers
                st.session_state["quiz_done"]    = True
                st.session_state["pending_quiz"] = None
                # Re-run agent with answers to get full career guidance
                with st.spinner("🔍 Analysing your aptitude results…"):
                    last_query  = st.session_state["messages"][-1]["content"] if st.session_state["messages"] else "Career guidance"
                    result      = run_agent(last_query, st.session_state["profile"])
                    final_resp  = result.get("final_response", "Sorry, I couldn't generate a response.")
                    sources     = result.get("sources", [])

                with st.chat_message("assistant"):
                    st.markdown(final_resp)
                    _render_sources_expander(sources)

                st.session_state["messages"].append({"role": "assistant", "content": final_resp})
                _render_download_button()
                st.rerun()
        return   # wait for quiz submission

    # Chat input
    if user_input := st.chat_input("Ask me anything about studies, careers, admissions…"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking…"):
                result     = run_agent(user_input, st.session_state["profile"])
                final_resp = result.get("final_response", "Sorry, I couldn't generate a response.")
                sources    = result.get("sources", [])
                category   = result.get("category", "")
                ease_score = result.get("ease_score_text", "")

            # Stream-display (simulate streaming with markdown)
            st.markdown(final_resp)

            # Ease score as expander if abroad
            if ease_score:
                with st.expander("🌍 Visa + Admission Ease Score Details"):
                    st.markdown(ease_score)

            _render_sources_expander(sources)

        st.session_state["messages"].append({"role": "assistant", "content": final_resp})

        # Check if quiz was queued
        if st.session_state.get("pending_quiz"):
            st.rerun()
        else:
            _render_download_button()


def _render_sources_expander(sources: List[str]):
    if sources:
        with st.expander("📚 Sources cited"):
            for s in sources:
                st.markdown(f"- `{s}`")


def _render_download_button():
    pdf_path = st.session_state.get("last_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Session Record (PDF)",
                data=f,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK-START PROMPTS (shown when chat is empty)
# ══════════════════════════════════════════════════════════════════════════════

STARTER_PROMPTS = [
    "What stream should I choose after Class 10 if I love computers?",
    "I scored 85% in Class 12 PCM. Which IITs/NITs can I target?",
    "Help me plan for studying in Canada after my B.Tech.",
    "I'm feeling stressed about board exams. How do I manage it?",
    "What are the best MBA colleges in India for a Commerce student?",
    "Explain the IELTS vs TOEFL difference for UK universities.",
]


def render_starters():
    if not st.session_state["messages"]:
        st.markdown("### 👋 Welcome! Try one of these:")
        cols = st.columns(2)
        for i, prompt in enumerate(STARTER_PROMPTS):
            col = cols[i % 2]
            if col.button(f"💬 {prompt}", key=f"starter_{i}", use_container_width=True):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="CounsellAI – Education Counsellor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global CSS
    st.markdown("""
    <style>
    /* Brand colours */
    :root {
        --brand-blue:   #2563EB;
        --brand-light:  #EFF6FF;
        --accent-green: #16A34A;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px 8px;
    }
    /* Download button */
    .stDownloadButton > button {
        background-color: var(--accent-green) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: var(--brand-blue) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--brand-light);
    }
    /* Title */
    h1 { color: var(--brand-blue); }
    </style>
    """, unsafe_allow_html=True)

    init_session()
    render_sidebar()

    # Main column layout
    main_col, info_col = st.columns([3, 1])

    with main_col:
        render_starters()
        render_chat()

    with info_col:
        st.markdown("### ℹ️ How I Can Help")
        st.markdown("""
**📚 Academic**
Study plans, subject choices, board exam tips

**💼 Career**
Aptitude assessment, career paths, entrance exams

**🌍 Admissions**
College shortlisting, applications, scholarships

**❤️ Personal**
Stress management, motivation, parental pressure

**📄 Documentation**
Every session is saved as a PDF
        """)

        st.divider()
        st.markdown("### 🔒 Safety Note")
        st.info(
            "For serious personal or mental health concerns, "
            "please reach out to **iCall**: 9152987821 or **Vandrevala Foundation**: 1860-2662-345"
        )

        st.divider()
        st.markdown("### 📊 Session Stats")
        n_turns = len([m for m in st.session_state["messages"] if m["role"] == "user"])
        st.metric("Questions asked", n_turns)


if __name__ == "__main__":
    main()
