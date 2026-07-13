"""
orchestrator.py — Plan -> Execute -> Reflect -> Synthesize loop.

This replaces the flat single-shot agent loop (agent.build_agent) with a
smarter structure:

  1. PLAN       - break the research goal into focused sub-questions.
  2. EXECUTE    - answer each sub-question with a scoped subtask agent
                  (agent.build_subtask_agent), recording exactly which
                  URLs it actually used.
  3. REFLECT    - check whether the combined findings actually cover the
                  original goal; if there's an obvious gap, run one more
                  targeted subtask to close it. This is the "real stopping
                  condition" instead of just looping until max_iterations.
  4. SYNTHESIZE - write the final report with inline [1], [2] citation
                  markers. The numbered source list is built HERE, in code,
                  from the URLs actually seen during execution -- never
                  trusted to the LLM to invent. A validation pass afterward
                  strips any citation number the model made up that doesn't
                  correspond to a real source.

Public interface (this is what streamlit_app.py / eval.py call):

    run_research_agent(goal: str, callbacks=None, max_iterations: int = 10) -> str

Returns the final markdown report as a string, with a code-verified
"Sources" section appended.
"""

import json
import re

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_groq import ChatGroq

from agent import build_subtask_agent

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PLANNER_MODEL = "llama-3.1-8b-instant"
SYNTHESIS_MODEL = "llama-3.1-8b-instant"

MAX_SUBQUESTIONS = 4
URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+")
CITATION_RE = re.compile(r"\[(\d+)\]")


PLANNER_SYSTEM_PROMPT = """You break a research goal into focused sub-questions
that, together, fully cover the goal. Rules:
- Produce between 2 and 4 sub-questions. Fewer, well-scoped sub-questions are
  better than many overlapping ones.
- Each sub-question should be answerable independently via web research.
- Do not include a sub-question that just restates the whole goal.

Respond with ONLY a JSON array of strings, no preamble, no markdown fences.
Example: ["What is X?", "How does X compare to Y?"]
"""

REFLECTION_SYSTEM_PROMPT = """You check whether a set of research findings
fully covers an original research goal. You will be given the goal and the
findings gathered so far.

If the findings already cover the goal well, respond with exactly: NONE
If there is one clear, specific gap worth one more targeted search, respond
with a single follow-up sub-question that would close that gap -- just the
question text, nothing else.

Respond with ONLY "NONE" or ONLY the follow-up question. No preamble.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are writing the final report for a research
goal, using findings already gathered by a research team. You will be given
the goal, each finding, and the numbered list of sources that back each
finding.

Rules:
- Structure the report with clear headers and, where relevant, a comparison
  table. Every table cell must contain a specific, concrete value or short
  note -- never leave a cell blank.
- Every non-trivial claim must end with an inline citation marker like [1]
  or [2] referencing the EXACT source number it came from, taken only from
  the source numbers provided to you. Do not invent source numbers.
- If a claim isn't backed by any specific source, don't attach a citation
  to it -- write it as general context instead of citing a source you
  weren't given.
- Do NOT write your own "Sources" section -- that is appended separately
  in code after your report.
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _extract_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
    return []


def _urls_from_intermediate_steps(intermediate_steps) -> list:
    """Pull every URL actually touched by tool calls during a subtask run."""
    urls = []
    for action, observation in intermediate_steps or []:
        tool_input = getattr(action, "tool_input", None)
        if isinstance(tool_input, dict):
            for v in tool_input.values():
                if isinstance(v, str):
                    urls.extend(URL_RE.findall(v))
        elif isinstance(tool_input, str):
            urls.extend(URL_RE.findall(tool_input))

        if isinstance(observation, str):
            urls.extend(URL_RE.findall(observation))

    # de-dupe, preserve order
    seen = set()
    ordered = []
    for u in urls:
        u = u.rstrip(".,;")
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _plan(goal: str) -> list:
    llm = ChatGroq(model=PLANNER_MODEL, temperature=0)
    messages = [
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", f"Research goal:\n{goal}"),
    ]
    response = llm.invoke(messages)
    subquestions = _extract_json_array(response.content)
    if not subquestions:
        # planner failed to produce usable JSON -- fall back to treating
        # the whole goal as a single sub-question rather than crashing
        return [goal]
    return subquestions[:MAX_SUBQUESTIONS]


def _reflect(goal: str, findings: list) -> str | None:
    llm = ChatGroq(model=PLANNER_MODEL, temperature=0)
    findings_text = "\n\n".join(f"- {q}\n  {a}" for q, a, _ in findings)
    messages = [
        ("system", REFLECTION_SYSTEM_PROMPT),
        ("human", f"Original goal:\n{goal}\n\nFindings so far:\n{findings_text}"),
    ]
    response = llm.invoke(messages)
    text = response.content.strip()
    if text.upper().startswith("NONE"):
        return None
    return text


def _run_subtask(question: str, callbacks, max_iterations: int):
    executor = build_subtask_agent(
        callbacks=callbacks,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
    )
    try:
        result = executor.invoke({"input": question})
    except Exception as e:
        return f"(This sub-question could not be answered: {e})", []

    answer = result.get("output", "").strip()
    steps = result.get("intermediate_steps", [])
    urls = _urls_from_intermediate_steps(steps)
    return answer, urls


def _synthesize(goal: str, findings: list, source_map: dict) -> str:
    """
    findings: list of (question, answer, [urls]) tuples
    source_map: url -> index (1-based), already built in code
    """
    findings_block_parts = []
    for question, answer, urls in findings:
        source_nums = [source_map[u] for u in urls if u in source_map]
        source_nums_str = ", ".join(f"[{n}]" for n in sorted(set(source_nums))) or "(no specific source)"
        findings_block_parts.append(
            f"Sub-question: {question}\nFinding: {answer}\nAvailable sources for this finding: {source_nums_str}"
        )
    findings_block = "\n\n".join(findings_block_parts)

    llm = ChatGroq(model=SYNTHESIS_MODEL, temperature=0)
    messages = [
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        (
            "human",
            f"Research goal:\n{goal}\n\nFindings and their available sources:\n\n{findings_block}",
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _validate_and_finalize_citations(report: str, source_map: dict) -> tuple[str, dict]:
    """
    Strip any citation marker [n] that doesn't correspond to a real source
    (i.e. the model invented it), and append a code-built Sources section.
    Returns (final_report, stats).
    """
    index_to_url = {i: u for u, i in source_map.items()}
    max_index = len(source_map)

    found = [int(n) for n in CITATION_RE.findall(report)]
    valid_citations = [n for n in found if 1 <= n <= max_index]
    invalid_citations = [n for n in found if not (1 <= n <= max_index)]

    def _strip_invalid(match: re.Match) -> str:
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= max_index else ""

    cleaned_report = CITATION_RE.sub(_strip_invalid, report)

    sources_used = sorted(set(valid_citations))
    if sources_used:
        lines = ["\n\n## Sources\n"]
        for n in sources_used:
            lines.append(f"[{n}] {index_to_url[n]}")
        cleaned_report = cleaned_report.rstrip() + "\n" + "\n".join(lines)
    elif source_map:
        # sources exist but nothing got cited -- still list them so nothing
        # gets silently lost
        lines = ["\n\n## Sources (referenced during research, not cited inline)\n"]
        for u, n in source_map.items():
            lines.append(f"[{n}] {u}")
        cleaned_report = cleaned_report.rstrip() + "\n" + "\n".join(lines)

    stats = {
        "total_sources_found": max_index,
        "citations_used": len(sources_used),
        "invalid_citations_removed": len(set(invalid_citations)),
    }
    return cleaned_report, stats


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def run_research_agent(goal: str, callbacks=None, max_iterations: int = 10) -> str:
    """
    Plan -> Execute -> Reflect -> Synthesize, with code-verified citations.

    goal: the research goal from the user
    callbacks: LangChain callback handlers (e.g. the Streamlit step display)
    max_iterations: per-subtask tool-call budget
    """
    subquestions = _plan(goal)

    findings = []  # list of (question, answer, [urls])
    for question in subquestions:
        answer, urls = _run_subtask(question, callbacks, max_iterations)
        findings.append((question, answer, urls))

    # Reflection: at most one extra round, to keep this bounded and fast
    follow_up = _reflect(goal, findings)
    if follow_up:
        answer, urls = _run_subtask(follow_up, callbacks, max_iterations)
        findings.append((follow_up, answer, urls))

    # Build the code-owned, de-duplicated source map across ALL findings
    source_map = {}
    for _, _, urls in findings:
        for u in urls:
            if u not in source_map:
                source_map[u] = len(source_map) + 1

    report = _synthesize(goal, findings, source_map)
    final_report, stats = _validate_and_finalize_citations(report, source_map)

    if callbacks:
        for cb in callbacks:
            container = getattr(cb, "container", None)
            if container is not None:
                container.markdown(
                    f"📎 **Citations:** {stats['citations_used']} inline citation(s) "
                    f"across {stats['total_sources_found']} source(s) gathered"
                    + (
                        f" — removed {stats['invalid_citations_removed']} invalid citation(s)"
                        if stats["invalid_citations_removed"]
                        else ""
                    )
                )

    return final_report