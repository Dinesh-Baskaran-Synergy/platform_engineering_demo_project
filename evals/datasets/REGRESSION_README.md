# Regression cases

`regression.jsonl` holds one case per bug that reached users (or nearly did). The rule:
**every production bug gets a test before it gets a fix**, so it can never silently come back.

## How to add a regression case

1. Reproduce the bug and write down the exact input (or multi-turn `messages`).
2. Append one JSON line to `regression.jsonl`:
   ```json
   {"id": "rg-004", "category": "regression", "task": "chat", "input": "<exact input>",
    "expected": {}, "graders": [{"type": "not_contains", "values": ["<the bad output>"]}],
    "tags": ["bug:TICKET-123"], "severity": "critical",
    "notes": "What went wrong, when, and the root cause."}
   ```
   * Use the cheapest grader that pins the bug (`not_contains`, `regex`, `tool_call_match`, ...).
     Add an `llm_judge` grader only if the failure is about quality, not a concrete string.
   * Tag it with the ticket id (`bug:XYZ-1`) so the report links back to the incident.
   * Regression cases are usually `critical`: the category threshold is 100%.
3. Run it alone: `python -m evals.runner --ids rg-004` — it should FAIL before the fix and PASS after.
4. Run the full suite, then `make baseline` and commit `regression.jsonl` + `results/baseline.json`.
