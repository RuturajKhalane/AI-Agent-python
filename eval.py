"""
eval.py — Automated evaluation harness for the research agent.

Runs a fixed test set of research goals through build_agent() and records,
for each run:
  - wall-clock latency
  - whether it completed (vs. hit max_iterations / max_execution_time)
  - number of tool calls, broken down by tool
  - number of distinct source URLs touched (web_search results + scrape_page URLs)
  - number of tool-level errors encountered (failed fetches, empty searches, etc.)

What this script CANNOT automate: whether the final report is actually
*correct* and *complete*. That requires a human to read the report and
compare it against the sources. This script writes out a full transcript
per query (goal, report, every tool call + result) into eval_results/raw/
so you can do that scoring pass — see SCORING_GUIDE.md for the rubric.

Usage:
    python eval.py

Outputs:
    eval_results/raw/<NN>_<slug>.md   — one file per query, full transcript
    eval_results/results.csv          — automated metrics + empty columns
                                         for manual accuracy/completeness scores
"""

import csv
import os
import re
import time
from pathlib import Path

from agent import build_agent

OUT_DIR = Path("eval_results")
RAW_DIR = OUT_DIR / "raw"

URL_RE = re.compile(r"https?://[^\s\)\]\"']+")

ERROR_MARKERS = (
    "error",
    "failed",
    "search failed",
    "no results found",
    "returned no extractable text",
)

MAX_ITER_MARKERS = (
    "agent stopped due to max iterations",
    "agent stopped due to iteration limit",
)

# --- Test set: 18 queries across three categories -----------------------
# category is used only for grouping in the summary; it doesn't change
# how the query is run.
TEST_QUERIES = [
    # -- Factual (single, checkable answer) --
    ("factual", "What is the current stable version of Python?"),
    ("factual", "Who is the current CEO of OpenAI?"),
    ("factual", "What is the population of Japan as of the most recent estimate?"),
    ("factual", "What year was the first iPhone released?"),
    ("factual", "What is the boiling point of water at sea level in Celsius?"),
    # -- Comparative (requires synthesizing multiple sources) --
    ("comparative", "Compare AWS, Azure, and GCP pricing for a small startup's first year."),
    ("comparative", "Compare React and Vue for building an admin dashboard."),
    ("comparative", "Compare the top 5 SQL query optimization techniques across MySQL, PostgreSQL, and Oracle."),
    ("comparative", "Compare Python and JavaScript for backend web development."),
    ("comparative", "Compare the Tesla Model 3 and Model Y in terms of price and range."),
    ("comparative", "Compare the pros and cons of remote work versus in-office work for software teams."),
    ("comparative", "Compare Notion and Obsidian for personal knowledge management."),
    # -- Ambiguous / tricky (vague scope, no single correct answer) --
    ("ambiguous", "Research the best programming language to learn in 2026."),
    ("ambiguous", "Research the current state of AI safety efforts."),
    ("ambiguous", "Which is better, iOS or Android?"),
    ("ambiguous", "Research the future of remote work."),
    ("ambiguous", "Evaluate whether remote learning is effective."),
    ("ambiguous", "Research renewable energy adoption trends in rural India."),
]


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


def extract_urls(text: str) -> set:
    return set(URL_RE.findall(text or ""))


def looks_like_error(observation: str) -> bool:
    obs_lower = (observation or "").lower()
    return any(marker in obs_lower for marker in ERROR_MARKERS)


def run_single_eval(index: int, category: str, goal: str) -> dict:
    print(f"\n{'=' * 70}\n[{index:02d}] ({category}) {goal}\n{'=' * 70}")

    executor = build_agent(max_iterations=10, return_intermediate_steps=True)

    start = time.perf_counter()
    error_message = ""
    try:
        result = executor.invoke({"input": goal})
        report = result.get("output", "")
        steps = result.get("intermediate_steps", [])
    except Exception as e:
        report = ""
        steps = []
        error_message = str(e)
    latency = time.perf_counter() - start

    # --- derive metrics from intermediate_steps ---
    tool_call_count = len(steps)
    tool_counts = {"web_search": 0, "scrape_page": 0, "calculator": 0}
    all_urls = set()
    tool_errors = 0
    transcript_lines = []

    for action, observation in steps:
        tool_name = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", "")
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        all_urls |= extract_urls(str(tool_input))
        all_urls |= extract_urls(str(observation))

        if looks_like_error(str(observation)):
            tool_errors += 1

        transcript_lines.append(
            f"### Tool call: `{tool_name}`\n"
            f"**Input:** `{tool_input}`\n\n"
            f"**Observation:**\n```\n{str(observation)[:1500]}\n```\n"
        )

    hit_iteration_limit = any(marker in report.lower() for marker in MAX_ITER_MARKERS)
    completed = bool(report) and not hit_iteration_limit and not error_message

    # --- write transcript file ---
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = RAW_DIR / f"{index:02d}_{slugify(goal)}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"# Query {index:02d} ({category})\n\n")
        f.write(f"**Goal:** {goal}\n\n")
        f.write(f"**Latency:** {latency:.1f}s  \n")
        f.write(f"**Completed:** {completed}  \n")
        f.write(f"**Tool calls:** {tool_call_count} "
                f"(search={tool_counts.get('web_search', 0)}, "
                f"scrape={tool_counts.get('scrape_page', 0)}, "
                f"calc={tool_counts.get('calculator', 0)})  \n")
        f.write(f"**Distinct source URLs:** {len(all_urls)}  \n")
        f.write(f"**Tool-level errors encountered:** {tool_errors}  \n")
        if error_message:
            f.write(f"**Run-level exception:** {error_message}  \n")
        f.write("\n---\n\n## Final report\n\n")
        f.write(report if report else "_(no report produced)_")
        f.write("\n\n---\n\n## Tool call trace\n\n")
        f.write("\n".join(transcript_lines) if transcript_lines else "_(no tool calls recorded)_")
        f.write("\n\n---\n\n## Manual scoring (fill in after reading the report above)\n\n")
        f.write("- **Answered the question (1-5):** \n")
        f.write("- **Factual accuracy (1-5):** \n")
        f.write("- **Notes:** \n")

    return {
        "index": index,
        "category": category,
        "goal": goal,
        "completed": completed,
        "latency_sec": round(latency, 1),
        "tool_call_count": tool_call_count,
        "search_calls": tool_counts.get("web_search", 0),
        "scrape_calls": tool_counts.get("scrape_page", 0),
        "calculator_calls": tool_counts.get("calculator", 0),
        "distinct_sources": len(all_urls),
        "tool_errors": tool_errors,
        "run_exception": error_message,
        "transcript_file": str(fname),
        "answered_score_1_5": "",   # filled in manually
        "accuracy_score_1_5": "",   # filled in manually
    }


def main():
    if not os.getenv("GROQ_API_KEY") or not os.getenv("SERPAPI_API_KEY"):
        print("ERROR: GROQ_API_KEY and SERPAPI_API_KEY must both be set (check your .env).")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (category, goal) in enumerate(TEST_QUERIES, start=1):
        row = run_single_eval(i, category, goal)
        rows.append(row)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    completed_count = sum(1 for r in rows if r["completed"])
    avg_latency = sum(r["latency_sec"] for r in rows) / len(rows)
    avg_sources = sum(r["distinct_sources"] for r in rows) / len(rows)

    print(f"\n\n{'=' * 70}")
    print("EVAL RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"Completed: {completed_count}/{len(rows)}")
    print(f"Avg latency: {avg_latency:.1f}s")
    print(f"Avg distinct sources per report: {avg_sources:.1f}")
    print(f"\nRaw transcripts: {RAW_DIR}/")
    print(f"Metrics CSV:     {csv_path}")
    print("\nNext step: open each file in eval_results/raw/, read the report,")
    print("and fill in the 'Manual scoring' section at the bottom of each.")
    print("Then transfer those scores into results.csv's answered_score_1_5")
    print("and accuracy_score_1_5 columns, and run summarize_eval.py.")


if __name__ == "__main__":
    main()
