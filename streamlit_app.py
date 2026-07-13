"""
Streamlit UI for the Autonomous Research Agent.
"""

import os
import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from orchestrator import run_research_agent


st.set_page_config(page_title="Autonomous Research Agent", page_icon="\U0001F50E", layout="wide")


# --- Simple, crash-proof step callback --------------------------------
# Replaces langchain_community's StreamlitCallbackHandler, which throws
# "Current LLMThought is unexpectedly None!" when used with tool-calling
# agents. This version just prints each tool call and result as plain
# markdown lines -- no fragile internal state to break.
class SimpleStepCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container

    def on_agent_action(self, action, **kwargs):
        self.container.markdown(f"🔧 **Using tool:** `{action.tool}` — `{action.tool_input}`")

    def on_tool_end(self, output, **kwargs):
        text = str(output)
        preview = text[:500] + ("..." if len(text) > 500 else "")
        self.container.markdown(f"📄 **Result:** {preview}")


# --- Purple theme -----------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary-color: #8b5cf6;
    }
    .stApp {
        background-color: #0f0a1a;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c1a52 0%, #1a0f2e 55%, #120a20 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 4px 0 24px rgba(139, 92, 246, 0.25);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0;
    }
    h1, h2, h3 {
        color: #c4b5fd;
    }
    .stButton>button {
        background-color: #8b5cf6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #a78bfa;
        color: white;
    }
    .stButton>button:disabled {
        background-color: #4c3a6b;
        color: #cbb9f0;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #ffffff;
        text-align: left;
        justify-content: flex-start;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #ffffff;
        border-color: #ffffff;
        color: #4c1d95;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #ffffff !important;
        color: #4c1d95 !important;
        border-color: #ffffff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #f0eaff !important;
        color: #4c1d95 !important;
    }
    div[data-baseweb="select"] > div {
        border-color: #8b5cf6 !important;
    }
    textarea, input {
        border-color: #8b5cf6 !important;
    }
    ::selection {
        background: #8b5cf6;
    }

    /* --- Sidebar polish (violet + white only) --- */
    [data-testid="stSidebar"] * {
        color: #f5f2ff;
    }
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sidebar-divider {
        height: 1px;
        margin: 1.5rem 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        border: none;
    }

    /* --- Sidebar header block --- */
    .sb-header {
        text-align: center;
        padding: 1.6rem 0.8rem 1.4rem 0.8rem;
        margin: -1rem -1rem 1.2rem -1rem;
        background: radial-gradient(circle at top, rgba(255,255,255,0.10), transparent 70%);
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }
    .sb-header .sb-badge {
        width: 52px;
        height: 52px;
        margin: 0 auto 0.6rem auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        background: #ffffff;
        box-shadow: 0 0 0 4px rgba(255,255,255,0.12), 0 0 24px rgba(255,255,255,0.35);
    }
    .sb-header .sb-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.4px;
    }
    .sb-header .sb-subtitle {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.65);
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* --- Sidebar section label --- */
    .sb-section-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 0 0 0.9rem 0;
    }
    .sb-section-label::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #ffffff;
        box-shadow: 0 0 8px 2px rgba(255,255,255,0.7);
    }

    /* --- Custom status cards (violet + white only) --- */
    .purple-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        color: #f5f2ff;
        font-size: 0.9rem;
    }
    .sb-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .sb-status-pill.ok {
        background: #ffffff;
        color: #4c1d95;
    }
    .sb-status-pill.warn {
        background: transparent;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.6);
    }
    .sb-status-pill .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }
    .sb-status-pill.ok .dot { background: #4c1d95; }
    .sb-status-pill.warn .dot { background: #ffffff; }
    .sb-footer-note {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.55);
        text-align: center;
        line-height: 1.5;
    }
    .sb-footer-note code {
        background: rgba(255,255,255,0.1);
        color: #ffffff;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
    }
    .sb-footer-note b {
        color: #ffffff;
    }

    /* --- Pointer cursor for selectbox --- */
    div[data-baseweb="select"], div[data-baseweb="select"] * {
        cursor: pointer !important;
    }

    /* --- Depth picker caption --- */
    .sb-depth-caption {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin: 0.5rem 0 0.2rem 0;
    }
    .sb-depth-caption b {
        color: #ffffff;
    }
    [data-testid="stSidebar"] [data-testid="column"] .stButton>button {
        padding: 0.45em 0.2em;
        text-align: center;
        justify-content: center;
        font-size: 0.72rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
    }
    [data-testid="stSidebar"] [data-testid="column"] {
        min-width: 0;
    }

    /* --- Session stats card --- */
    .sb-stats-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .sb-stats-card .sb-stats-num {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .sb-stats-card .sb-stats-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* --- Checkboxes white --- */
    [data-testid="stCheckbox"] svg {
        fill: #ffffff !important;
    }

    /* --- Expanders --- */
    [data-testid="stExpander"] {
        background: rgba(139, 92, 246, 0.06);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary {
        color: #c4b5fd !important;
        font-weight: 600;
    }

    /* --- File uploader --- */
    [data-testid="stFileUploader"] section {
        background: rgba(139, 92, 246, 0.05);
        border: 1px dashed #8b5cf6;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] button {
        background-color: #8b5cf6 !important;
        color: white !important;
        border: none !important;
    }

    /* --- Dropdown option hover --- */
    li[role="option"]:hover {
        background-color: rgba(139, 92, 246, 0.25) !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Map secrets -> env vars
for key in ("GROQ_API_KEY", "SERPAPI_API_KEY"):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = str(st.secrets[key])

st.title("\U0001F50E Autonomous Research Agent")
st.caption(
    "Give it a broad research goal. It plans its own steps, searches the web, "
    "scrapes pages for ground-truth detail, and compiles a structured report."
)

if "history" not in st.session_state:
    st.session_state.history = []
if "goal_input" not in st.session_state:
    st.session_state.goal_input = ""

with st.sidebar:
    # ============================================================
    # HEADER
    # ============================================================
    st.markdown(
        """
        <div class="sb-header">
            <div class="sb-badge">🔮</div>
            <div class="sb-title">Research Agent</div>
            <div class="sb-subtitle">Autonomous · Groq-powered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Status Section ---
    st.markdown("### 📌 Status")

    st.markdown(
        """
        <div style="
            background: #140924;
            border: 1px solid #a78bfa;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #ffffff;
            font-size: 0.9rem;">
            ✅ Ready to run research
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background: #140924;
            border: 1px solid #a78bfa;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #ffffff;
            font-size: 0.9rem;">
            ⚠️ API keys required
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background: #140924;
            border: 1px solid #a78bfa;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.7rem;
            color: #ffffff;
            font-size: 0.9rem;">
            📊 Reports saved in session
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Max Iterations Slider ---
    st.markdown(
        """
        <div style="margin-bottom:0.5rem; font-weight:600; color:#e3dfe8;">
            Max Iterations
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_iterations = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        key="max_iterations",
        label_visibility="collapsed",
    )

    # --- Tips Section ---
    st.markdown("### 💡 Tips")
    st.markdown(
        """
        - Break down large goals into smaller sub‑tasks for better results
        - Use consistent keywords to improve search accuracy
        - Save and review past reports to refine future research
        - Please be patient; the agent may take a few minutes to complete complex research tasks
        """
    )

    # --- Decorative Divider ---
    st.markdown("---")

    # --- Credits ---
    st.markdown("### 🔖 Credits")
    st.caption("Made with ❤️ using Streamlit + LangChain")


# --- Main content ---
if st.session_state.get("show_last_report") and st.session_state.history:
    last_goal, last_report = st.session_state.history[-1]
    st.markdown("### 📂 Last saved report")
    st.markdown(f"**Goal:** {last_goal}")
    st.markdown(last_report)
    st.markdown("---")

goal = st.text_area(
    "Research goal",
    key="goal_input",
    placeholder="e.g. Research the top 3 project management tools and compare their pricing tiers",
    height=80,
)

run_clicked = st.button("Run research", type="primary", disabled=not goal.strip())

if run_clicked:
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.getenv("SERPAPI_API_KEY"):
        missing.append("SERPAPI_API_KEY")

    if missing:
        st.markdown(
            f"""
            <div style="background: linear-gradient(145deg, #3b1f5c, #2a1445);
                        border: 1px solid #a78bfa; border-radius: 10px;
                        padding: 0.9rem 1.1rem; color: #f0e9ff;">
                ⚠️ Missing required secret(s): <b>{', '.join(missing)}</b>.
                Add them in the sidebar's secrets file and reload.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Agent reasoning")
        step_box = st.container()
        st_callback = SimpleStepCallback(step_box)

        report = None
        with st.spinner("Researching..."):
            try:
                report = run_research_agent(
                    goal, callbacks=[st_callback], max_iterations=max_iterations
                )
            except Exception as e:
                st.error(f"Agent run failed: {e}")

        if report:
            st.session_state.history.append((goal, report))
            st.markdown("### Final report")
            st.markdown(report)
            st.download_button(
                "Download report (.md)",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
            )

if st.session_state.history:
    with st.expander(f"Past runs in this session ({len(st.session_state.history)})"):
        for i, (g, r) in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"**{i}. {g}**")
            st.markdown(r)
            st.markdown("---")