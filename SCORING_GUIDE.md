# Evaluation Scoring Guide

`eval.py` automates everything that can be measured objectively: completion
rate, latency, tool-call counts, source counts, and tool-level errors. It
cannot automate whether a report is actually *correct* — that requires a
human to read it and check it against real-world facts. This guide is the
rubric for that manual pass.

## How to score each report

After running `python eval.py`, open each file in `eval_results/raw/`.
Every file ends with a "Manual scoring" section — fill in these two scores
for each one:

### 1. Answered the question (1–5)
Does the report actually address what was asked, at the scope implied by
the query?

| Score | Meaning |
|---|---|
| 5 | Directly and completely answers the question as asked |
| 4 | Answers the question but misses a minor part of the scope |
| 3 | Partially answers — covers the general topic but dodges the specific ask |
| 2 | Mostly off-target; touches the topic but not the actual question |
| 1 | Does not answer the question at all, or the run failed to produce a report |

### 2. Factual accuracy (1–5)
Spot-check 2–3 concrete claims in the report against the source URLs listed
in the tool call trace (open a couple of them). Ask: does the report
represent what the sources actually say, or does it distort/invent details?

| Score | Meaning |
|---|---|
| 5 | All checked claims match their sources; no fabrication |
| 4 | Minor imprecision (e.g. rounding, mild overgeneralization) but no wrong facts |
| 3 | At least one claim is unsupported by any cited source, but not clearly false |
| 2 | At least one claim contradicts its cited source or is clearly outdated/wrong |
| 1 | Report is substantially fabricated or contradicted by its own sources |

## Notes field
Use the free-text "Notes" line for anything the numeric scores don't
capture — e.g. "cited a source but never actually scraped it, just
summarized the search snippet," or "correct but repetitive across sections."
Interesting failure cases here are worth pulling into the README as
concrete examples of the agent's limits — that's more convincing than the
raw numbers alone.

## After scoring

1. Transfer your 1–5 scores into `eval_results/results.csv`, in the
   `answered_score_1_5` and `accuracy_score_1_5` columns (one row per
   query, matched by `index`).
2. Run:
   ```bash
   python summarize_eval.py
   ```
   This produces `eval_results/eval_results.md` — a clean summary block
   with the honest numbers, ready to paste into your README.

## Why this two-phase approach

A fully-automated "accuracy" score without ground truth would just be
another LLM call grading the agent's own work — which mostly measures
whether the report *sounds* confident, not whether it's *correct*. Doing
the check by hand, even on a small test set, is what turns "the agent
produced 18 reports" into an actual reliability claim you can defend in an
interview.
