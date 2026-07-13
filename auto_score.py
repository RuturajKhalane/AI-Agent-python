"""
auto_score.py — LLM-as-judge automated scoring for eval_results/raw/ transcripts.

This is a faster, automated alternative to the manual scoring pass described
in SCORING_GUIDE.md. Be clear-eyed about what this trades away:

  MANUAL scoring (a human reads the report and spot-checks sources) is
  stronger evidence for a portfolio/interview claim, because a human can
  catch subtle misrepresentations an LLM judge might rubber-stamp.

  This automated version has one thing going for it that a naive
  "ask an LLM to grade its own homework" setup doesn't: it grades the
  report against the ACTUAL tool call trace captured during that specific
  run (real search snippets and scraped text, saved in each transcript
  file) rather than against the judge's own general knowledge. That makes
  it a real "does the report's claims match what was actually retrieved"
  check, not just a fluency/confidence check.

  Still: treat these scores as a fast first pass, not a final verdict.
  Spot-check 3-4 of the 18 by hand afterward (5-10 min) to confirm the
  judge isn't systematically too generous or too harsh — and mention in
  your README that scoring was LLM-assisted with manual spot-checks, not
  purely manual, for an honest presentation.

Usage:
    python auto_score.py
"""

import csv
import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_groq import ChatGroq

RAW_DIR = Path("eval_results/raw")
RESULTS_CSV = Path("eval_results/results.csv")

JUDGE_SYSTEM_PROMPT = """You are grading a research agent's report. You will be given:
- The original research goal
- The agent's final report
- The tool call trace: the actual search results and scraped page content the
  agent retrieved while researching (this is the ground truth evidence available
  to it — judge the report against THIS, not your own general knowledge)

Score two things:
1. "answered_score" (1-5): Does the report directly and completely answer the
   goal, at the scope implied by the question?
   5 = fully answers; 3 = partially, dodges the specific ask; 1 = does not
   answer at all or no report was produced.
2. "accuracy_score" (1-5): Do the report's claims match what's actually in the
   tool call trace evidence? Penalize claims that appear nowhere in the
   retrieved evidence (likely fabricated/hallucinated), not claims that are
   simply general knowledge consistent with the evidence.
   5 = all checkable claims are supported by the trace; 3 = at least one
   claim has no support in the trace but isn't clearly false; 1 = report
   substantially contradicts or ignores its own evidence.

Respond with ONLY a JSON object, no preamble, no markdown fences:
{"answered_score": <int 1-5>, "accuracy_score": <int 1-5>, "notes": "<one sentence>"}
"""


def parse_transcript(text: str) -> dict:
    def section(header, next_headers):
        pattern = rf"## {re.escape(header)}\n\n(.*?)(?=\n---\n\n## (?:{'|'.join(re.escape(h) for h in next_headers)})|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    goal_match = re.search(r"\*\*Goal:\*\* (.+)", text)
    goal = goal_match.group(1).strip() if goal_match else ""

    report = section("Final report", ["Tool call trace"])
    trace = section("Tool call trace", ["Manual scoring"])

    return {"goal": goal, "report": report, "trace": trace}


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_one(llm: ChatGroq, goal: str, report: str, trace: str) -> dict:
    # trace can be long across many tool calls; cap it so the judge call stays fast/cheap
    trace_capped = trace[:6000]
    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", f"GOAL:\n{goal}\n\nFINAL REPORT:\n{report}\n\nTOOL CALL TRACE (evidence):\n{trace_capped}"),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response.content)
    if isinstance(parsed, dict) and "answered_score" in parsed and "accuracy_score" in parsed:
        return {
            "answered_score_1_5": parsed["answered_score"],
            "accuracy_score_1_5": parsed["accuracy_score"],
            "judge_notes": parsed.get("notes", ""),
        }
    return {"answered_score_1_5": "", "accuracy_score_1_5": "", "judge_notes": "(judge response could not be parsed)"}


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run eval.py first.")
        return
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run eval.py first.")
        return

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    if "judge_notes" not in fieldnames:
        fieldnames.append("judge_notes")

    files = sorted(RAW_DIR.glob("*.md"))
    file_by_index = {}
    for fp in files:
        match = re.match(r"(\d+)_", fp.name)
        if match:
            file_by_index[int(match.group(1))] = fp

    for row in rows:
        idx = int(row["index"])
        fp = file_by_index.get(idx)
        if not fp:
            print(f"[{idx:02d}] WARNING: no transcript file found, skipping")
            continue

        text = fp.read_text(encoding="utf-8")
        parsed = parse_transcript(text)

        if not parsed["report"] or "no report produced" in parsed["report"].lower():
            print(f"[{idx:02d}] No report produced — scoring as 1/1")
            row["answered_score_1_5"] = 1
            row["accuracy_score_1_5"] = 1
            row["judge_notes"] = "No report was produced for this run."
            continue

        print(f"[{idx:02d}] Judging: {row['goal'][:60]}...")
        scores = judge_one(llm, parsed["goal"], parsed["report"], parsed["trace"])
        row["answered_score_1_5"] = scores["answered_score_1_5"]
        row["accuracy_score_1_5"] = scores["accuracy_score_1_5"]
        row["judge_notes"] = scores["judge_notes"]
        print(f"       answered={scores['answered_score_1_5']} accuracy={scores['accuracy_score_1_5']}")

        # also append the judge's notes into the transcript file itself,
        # clearly labeled as automated, so the file stays self-contained
        labeled_note = (
            f"\n\n---\n\n## LLM-judge scoring (automated, not human-reviewed)\n\n"
            f"- **Answered the question (1-5):** {scores['answered_score_1_5']}\n"
            f"- **Factual accuracy (1-5):** {scores['accuracy_score_1_5']}\n"
            f"- **Judge notes:** {scores['judge_notes']}\n"
        )
        fp.write_text(text + labeled_note, encoding="utf-8")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Scores written to {RESULTS_CSV} and appended to each transcript in {RAW_DIR}/")
    print("\nIMPORTANT: these are LLM-judge scores, not human-reviewed scores.")
    print("Before treating them as final, spot-check 3-4 transcripts by hand")
    print("(see SCORING_GUIDE.md) to confirm the judge is calibrated reasonably.")
    print("\nNext: python summarize_eval.py")


if __name__ == "__main__":
    main()