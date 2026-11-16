# CLAUDE.md

AcmeCloud Support Assistant: a sample LLM app plus a home-grown eval harness. The app is small on purpose; **the evals are the product**.
The README has the full guide. This file is the quick orientation.

## Quick start

```bash
make setup     # .venv + pinned deps, creates .env (mock mode)
make test      # 93 pytest tests, no network (sockets blocked in tests/conftest.py)
make eval      # 138 eval cases in mock mode; exit 1 if thresholds fail
make compare   # diff results/latest.json vs results/baseline.json; exit 1 on regression
```

Live: fill `.env`, then `make doctor`, then `make eval-live`. Compare models with `make models MODELS=a,b`. Check the judge with `make calibrate`.

## Tech stack

- Python 3.11+ (CI uses 3.11), venv + `requirements.txt` (exact pins; `litellm==1.109.1`).
- **LiteLLM is the only LLM interface** (`litellm.completion`). Never import provider SDKs.
- pydantic (schemas), jsonschema, rich (CLI tables), PyYAML, python-dotenv, pytest.
- Retrieval: hand-written BM25 plus char-trigram BM25. No vector DB and no eval framework.

## Architecture

```
user msg -> Assistant.respond (app/assistant.py)
  1. redact PII (app/privacy.py), truncate long input (keeps head + tail)
  2. route: follow-ups merge the previous user turn into the retrieval query
  3. retrieve top-k KB docs (app/retriever.py), injected as <documents> = DATA, not instructions
  4. LLM call with OpenAI-format tools (app/tools.py); loop runs tools, max 3 rounds.
     No native tool support -> prompt-based JSON tool protocol (prompts/tool_fallback.v1.md)
  5. redact PII in the answer -> AssistantResponse (answer, retrieved ids, tool calls, per-call metrics)
Extractor (app/extractor.py): email -> {intent, urgency, product, order_id|null, sentiment}
  response_format JSON schema if supported, else prompt JSON; pydantic validation + 1 repair retry
```

**LLM layer** (`app/llm_client.py`):
- `LLMConfig.from_env()` reads everything from env. Model names never appear in code.
- `LLMClient` is the interface, with two implementations:
  - `LiteLLMClient` passes `api_base`/`api_key` when `LITELLM_API_BASE` is set (proxy); otherwise LiteLLM routes directly to the provider. It sets `num_retries`, `timeout` and `fallbacks=[FALLBACK_MODEL]`.
  - `MockLLMClient` calls `litellm.completion(mock_response=ModelResponse)` with output from `app/mock_llm.py` plus `app/mock_fixtures/rules.yaml`. It is deterministic and offline.
- Every call returns an `LLMResult` with content, tool_calls, latency, tokens and cost. Cost is `None` when LiteLLM can't price the model; the runner then estimates it from tokens.
- `litellm.drop_params = True`. `LITELLM_LOCAL_MODEL_COST_MAP=True` stops LiteLLM fetching its price map over the network at import.

**Eval harness** (`evals/`):
- Each JSONL case (`schema.py: EvalCase`) has a `task`: `chat` | `extract` | `retrieve`.
- `runner.run_case` executes the case, then every grader spec, through `graders/registry.py`.
- `apply_consistency` adds a group check across paraphrases. `summarize` applies `config.yaml` thresholds, and any critical failure fails the gate.
- `report.py` writes `results/<run_id>.json`, `latest.json` and `report.md`. With `--models` it also writes `model_comparison.md`.

## File map

| Path | What |
|---|---|
| `app/assistant.py` | orchestration |
| `app/llm_client.py` | config, clients, cost/latency |
| `app/mock_llm.py`, `app/mock_fixtures/rules.yaml` | simulated model: canned rules first, then IDF-weighted KB sentence matching, tool decisions, heuristic extraction and judge |
| `app/retriever.py`, `app/text_utils.py` | BM25, stemmer, synonyms, fuzzy match |
| `app/tools.py` | `get_order_status`, `create_ticket`, `MockDB` (`ORD-9999` simulates an outage) |
| `app/extractor.py` | `TicketFields` schema + extraction |
| `app/privacy.py` | `redact_pii` (cards, CVV, passwords, SSN, keys), `redact_secrets` (logs/errors) |
| `app/doctor.py` | env check (never prints secrets) + 1 tiny call per model |
| `app/prompts/*.vN.md` | versioned prompts; version id `name@vN#sha8` is recorded in results |
| `kb/*.md` | 14 docs. `community-tips.md` contains a **deliberate** prompt injection; don't "fix" it |
| `evals/datasets/*.jsonl` | 17 categories, 138 cases (+ `REGRESSION_README.md`) |
| `evals/graders/` | `code_graders.py` (pure `check_*` plus `grade_*` adapters), `retrieval_metrics.py`, `llm_judge.py`, `registry.py` |
| `evals/rubrics/*.yaml` | judge rubrics (support_quality, faithfulness, tone, safety) |
| `evals/config.yaml` | per-category thresholds, overall gate, fallback pricing, retrieval k |
| `evals/runner.py`, `report.py`, `compare.py` | CLI, reporting, baseline diff |
| `evals/judge_calibration/` | 20 human-labelled examples + `calibrate.py` (agreement, kappa, MAE) |
| `results/baseline.json` | committed **mock-mode** baseline; other `results/*` files are git-ignored |
| `.github/workflows/evals.yml` | PRs: pytest + mock evals + compare. Manual dispatch: live evals via repo secrets |

## Conventions

- **Add a case:** append a JSONL line to the category file, then `python -m evals.runner --ids <id>`, `make eval`, `make baseline`, and commit the baseline.
- **Add a grader:** write a pure `check_x` plus a `grade_x(spec, ctx)` adapter, register it in `registry.py`, and add tests in `tests/test_code_graders.py`.
- **Change a prompt:** add `name.v2.md` (the newest version wins). Pin an older one with `PROMPT_VERSION_<NAME>=1`.
- **Mock vs live:** mock mode validates the harness, not model quality. If you change mock heuristics, re-run `make eval` and update the baseline intentionally.
- **Known mock failures (intentional):** `ec-008` needs date arithmetic; `fc-008` asks for custom JSON keys.
- **Security:** no secrets in code or results. `.env` is git-ignored. Errors pass through `redact_secrets`. Tests must not touch the network.
- **Commits:** conventional style (`feat:`, `fix:`, `test:`, `ci:`, `docs:`, `chore:`).

## Key env vars

`EVAL_MODE` (mock|live), `LITELLM_API_BASE`, `LITELLM_API_KEY`, `APP_MODEL`, `JUDGE_MODEL` (use a different model from the app),
`EXTRACTOR_MODEL` (defaults to `APP_MODEL`), `FALLBACK_MODEL`, `TOOL_CALLING` / `STRUCTURED_OUTPUT` (auto|native/schema|prompt; use them to override capability detection for proxy aliases),
`LLM_TIMEOUT_SECONDS`, `LLM_NUM_RETRIES`.

## Recent changes

- Initial build, in order: scaffold, then LiteLLM client + app, datasets, graders, runner/report/compare/calibration, tests, CI, README.
- Mock model tuned (IDF scoring, last-turn weighting, title bonus, format-word filtering) so mock mode passes the gate. Current result: 136/138, gate PASSED.
- `make` targets bootstrap `.venv` automatically. Dependencies pinned to the tested versions.
- Repo renamed `My_repo` → `platform_engineering_demo_project`.
- A results dashboard was published as a private claude.ai artifact. It is a snapshot of one mock run and does not refresh itself; re-upload the CSVs after new runs.

## Status / next steps

- Live mode is not yet exercised against real models. Run `make doctor`, then `make eval-live`, and keep a separate live baseline (`results/baseline_live.json`).
- Re-run `make calibrate` with the real `JUDGE_MODEL` before trusting judge-graded categories.
