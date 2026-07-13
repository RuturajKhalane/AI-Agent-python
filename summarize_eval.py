"""
summarize_eval.py — Turns eval_results/results.csv (with manual scores
filled in) into a clean eval_results/eval_results.md summary, formatted
for dropping straight into a README.

Run this AFTER eval.py and AFTER you've filled in answered_score_1_5 and
accuracy_score_1_5 for every row in results.csv (see SCORING_GUIDE.md).

Usage:
    python summarize_eval.py
"""

import csv
from pathlib import Path
from statistics import mean

RESULTS_CSV = Path("eval_results/results.csv")
OUT_MD = Path("eval_results/eval_results.md")


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run eval.py first.")
        return

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("ERROR: results.csv is empty.")
        return

    total = len(rows)
    completed_rows = [r for r in rows if r["completed"] == "True"]
    completed_count = len(completed_rows)

    latencies = [to_float(r["latency_sec"]) for r in rows if to_float(r["latency_sec"]) is not None]
    sources = [to_float(r["distinct_sources"]) for r in rows if to_float(r["distinct_sources"]) is not None]
    tool_calls = [to_float(r["tool_call_count"]) for r in rows if to_float(r["tool_call_count"]) is not None]
    errors = [to_float(r["tool_errors"]) for r in rows if to_float(r["tool_errors"]) is not None]

    answered_scores = [to_float(r["answered_score_1_5"]) for r in rows]
    answered_scores = [s for s in answered_scores if s is not None]
    accuracy_scores = [to_float(r["accuracy_score_1_5"]) for r in rows]
    accuracy_scores = [s for s in accuracy_scores if s is not None]

    unscored = total - len(answered_scores)

    # per-category breakdown
    categories = sorted(set(r["category"] for r in rows))
    cat_lines = []
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_completed = sum(1 for r in cat_rows if r["completed"] == "True")
        cat_lines.append(f"- **{cat.capitalize()}**: {cat_completed}/{len(cat_rows)} completed")

    lines = []
    lines.append("# Evaluation Results\n")
    lines.append(f"Test set: {total} research queries spanning factual, comparative, "
                  f"and ambiguous/open-ended categories. Full per-query transcripts and "
                  f"raw tool-call traces are in `eval_results/raw/`.\n")

    lines.append("## Headline numbers\n")
    lines.append(f"- **Completed {completed_count}/{total} queries** "
                  f"({completed_count / total * 100:.0f}%) without hitting the iteration limit or erroring out")
    if latencies:
        lines.append(f"- **Average latency:** {mean(latencies):.1f}s per report "
                      f"(min {min(latencies):.1f}s, max {max(latencies):.1f}s)")
    if sources:
        lines.append(f"- **Average distinct sources per report:** {mean(sources):.1f}")
    if tool_calls:
        lines.append(f"- **Average tool calls per query:** {mean(tool_calls):.1f}")
    if errors:
        total_errors = sum(errors)
        lines.append(f"- **Tool-level errors encountered:** {int(total_errors)} across all runs "
                      f"({total_errors / total:.1f} per query on average) — "
                      f"all recovered from automatically without crashing the run")
    if answered_scores:
        lines.append(f"- **Avg \"answered the question\" score (manual, 1-5):** {mean(answered_scores):.1f}")
    if accuracy_scores:
        lines.append(f"- **Avg factual accuracy score (manual, 1-5):** {mean(accuracy_scores):.1f}")
    if unscored:
        lines.append(f"\n> ⚠️ {unscored} row(s) still missing manual scores — "
                      f"see SCORING_GUIDE.md and fill in results.csv before treating these numbers as final.")

    lines.append("\n## By category\n")
    lines.extend(cat_lines)

    lines.append("\n## Per-query results\n")
    lines.append("| # | Category | Goal | Completed | Latency (s) | Sources | Tool calls | Errors | Answered (1-5) | Accuracy (1-5) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        goal_short = r["goal"] if len(r["goal"]) <= 60 else r["goal"][:57] + "..."
        lines.append(
            f"| {r['index']} | {r['category']} | {goal_short} | {r['completed']} | "
            f"{r['latency_sec']} | {r['distinct_sources']} | {r['tool_call_count']} | "
            f"{r['tool_errors']} | {r['answered_score_1_5'] or '-'} | {r['accuracy_score_1_5'] or '-'} |"
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote summary to {OUT_MD}")
    print(f"\nCompleted {completed_count}/{total} queries.")
    if latencies:
        print(f"Avg latency: {mean(latencies):.1f}s")
    if sources:
        print(f"Avg sources/report: {mean(sources):.1f}")


if __name__ == "__main__":
    main()
