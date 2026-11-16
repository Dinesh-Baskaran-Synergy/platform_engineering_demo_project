# AcmeCloud Support Assistant: an eval suite you can read in an afternoon

A complete, runnable example of how to **evaluate an LLM application**. The app is a small
customer-support assistant for a fictional cloud company, *AcmeCloud*. The evals are the point:
138 cases in 17 categories, code graders, an LLM judge with a calibration set, retrieval
metrics, budgets, baselines and regression detection, and CI. Everything is built without an
eval framework, so you can see how each piece works.

* **LiteLLM is the only way this project calls LLMs** (`litellm.completion`). It works with a LiteLLM
  proxy/gateway, or routes directly to providers (Anthropic, OpenAI, Bedrock, …).
* **No model names are in the code.** They come from env vars (`APP_MODEL`, `JUDGE_MODEL`, …).
* **Mock mode is the default.** The whole pipeline, including CI, runs with no API key and no network.

```bash
make setup     # venv + deps, creates .env from .env.example (mock mode)
make test      # 93 unit tests for app code AND graders, no network
make eval      # full eval suite, deterministic mock mode
```

---

## Contents
1. [What are evals?](#1-what-are-evals)
2. [The app under test](#2-the-app-under-test)
3. [Project layout](#3-project-layout)
4. [Configure LiteLLM: proxy vs direct](#4-configure-litellm-proxy-vs-direct)
5. [Pick models with env vars](#5-pick-models-with-env-vars)
6. [`make doctor`](#6-make-doctor)
7. [Eval categories and the philosophy behind each](#7-eval-categories-and-the-philosophy-behind-each)
8. [Graders](#8-graders)
9. [Run live](#9-run-live)
10. [Compare models](#10-compare-models)
11. [Read the report](#11-read-the-report)
12. [Baselines and regressions](#12-baselines-and-regressions)
13. [Add a test case](#13-add-a-test-case)
14. [Add a grader](#14-add-a-grader)
15. [Judge calibration](#15-judge-calibration)
16. [CI](#16-ci)
17. [Mock mode: what it is and isn't](#17-mock-mode-what-it-is-and-isnt)
18. [Security notes](#18-security-notes)

---

## 1. What are evals?

Evals are **tests for behaviour you cannot assert with `==`**. An LLM app takes open-ended input
and produces probabilistic output, so instead of one unit test per function you keep a
**dataset of representative situations**. Each situation comes with **graders** that decide whether
the behaviour was acceptable. You run the whole set on every change and track the pass rate per category.

A good eval suite:

* **Covers the risks, not just the happy path.** Safety, privacy, injection, hallucination and
  regressions get their own categories, and their cases are marked `critical`, so one failure blocks the release.
* **Prefers cheap, deterministic graders.** Use `contains`, `regex`, `json_schema` and `tool_call_match`
  wherever they work. An LLM judge is only for things code can't check (tone, helpfulness,
  faithfulness), and the judge is itself measured against human labels.
* **Is versioned with the code.** Prompts are versioned files. Every result records the prompt version,
  app model, judge model and git commit. A committed baseline lets `compare` point to the exact
  case that regressed.
* **Is cheap to run.** Mock mode runs in CI on every PR. Live mode runs on demand.

## 2. The app under test

`app/assistant.py` runs **route → retrieve → tools → answer**:

| Capability | Where | Notes |
|---|---|---|
| RAG Q&A | `app/retriever.py`, `kb/*.md` | BM25 plus a down-weighted character-trigram BM25 (typo-tolerant). 14 short KB docs. Retrieved docs are wrapped in `<documents>` and marked as *data, not instructions*. |
| Tool use | `app/tools.py` | `get_order_status(order_id)` and `create_ticket(summary, priority)` use OpenAI-format schemas and a mock in-memory DB. `ORD-9999` simulates a backend outage. |
| Structured extraction | `app/extractor.py` | Turns an email into `{intent, urgency, product, order_id\|null, sentiment}`. It uses JSON-schema `response_format` when `litellm.supports_response_schema(model)` is true; otherwise it prompts for JSON. Output is always validated with pydantic, with one repair retry. |

Defense in depth: user input is PII-redacted (cards, CVV, passwords, SSNs, API keys) **before** it reaches
the model, and the answer is redacted again on the way out (`app/privacy.py`). One KB doc
(`kb/community-tips.md`) **deliberately contains an injected instruction**, to test indirect prompt injection.

If a model has no native tool calling, the assistant switches to a **prompt-based JSON tool protocol**
(`app/prompts/tool_fallback.v1.md`) and logs a warning.

## 3. Project layout

```
app/
  llm_client.py        LLMConfig, LLMClient interface, LiteLLMClient, MockLLMClient, cost + latency capture
  assistant.py         orchestration (route -> retrieve -> tools -> answer)
  retriever.py         BM25 over kb/
  tools.py             tool schemas + mock DB
  extractor.py         email -> JSON (response_format or prompt + pydantic repair)
  doctor.py            env check + live smoke test
  mock_llm.py          deterministic "model" used in mock mode
  mock_fixtures/       canned responses (rules.yaml)
  prompts/             versioned system prompts (*.v1.md)
kb/                    knowledge base
evals/
  datasets/*.jsonl     one file per category (+ REGRESSION_README.md)
  graders/             code_graders.py, llm_judge.py, retrieval_metrics.py, registry.py
  rubrics/*.yaml       judge rubrics
  config.yaml          thresholds, pricing fallback, retrieval k
  runner.py            CLI
  report.py            console + JSON + Markdown
  compare.py           diff vs baseline, exit 1 on regression
  judge_calibration/   20 human-labelled examples + calibrate.py
tests/                 pytest (no network: sockets are blocked in conftest.py)
results/baseline.json  committed baseline (mock mode)
```

## 4. Configure LiteLLM: proxy vs direct

Copy `.env.example` to `.env` (`make setup` does this for you). **Never commit `.env`**; it is git-ignored.

### a) Through a LiteLLM proxy / gateway (recommended for teams)

```dotenv
EVAL_MODE=live
LITELLM_API_BASE=https://your-litellm-proxy.example.com
LITELLM_API_KEY=<your virtual key>
APP_MODEL=openai/<proxy-alias-for-app>
JUDGE_MODEL=openai/<proxy-alias-for-judge>
```

When `LITELLM_API_BASE` is set, `api_base` and `api_key` are passed on **every** LiteLLM call. The `openai/`
prefix tells LiteLLM to talk the OpenAI protocol to your proxy. The part after it is the proxy's
**model alias** (`model_name` in the proxy config). Provider keys stay on the proxy host.
`litellm_config.example.yaml` is a sample config if you run your own proxy
(`litellm --config litellm_config.example.yaml`).

### b) Direct provider routing

Leave `LITELLM_API_BASE`/`LITELLM_API_KEY` empty. Set the provider's standard env vars and use LiteLLM model strings:

```dotenv
EVAL_MODE=live
APP_MODEL=anthropic/<model-id>      # needs ANTHROPIC_API_KEY
JUDGE_MODEL=openai/<model-id>       # needs OPENAI_API_KEY
# or bedrock/<model-id> with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME
```

### Resilience and portability settings (in `app/llm_client.py`)

* `litellm.drop_params = True`: params a model doesn't support (e.g. `temperature`) are dropped instead of raising.
* `num_retries` (`LLM_NUM_RETRIES`, default 3): LiteLLM retries 429/5xx with exponential backoff.
* `timeout` (`LLM_TIMEOUT_SECONDS`, default 60).
* `fallbacks=[FALLBACK_MODEL]` if `FALLBACK_MODEL` is set.
* Messages and tool schemas are in **OpenAI format**. LiteLLM translates them for each provider.
* **Cost**: `litellm.completion_cost(completion_response=...)`. Proxy aliases and custom models usually
  have no price, so cost is **estimated from token usage** with
  `fallback_pricing_usd_per_1k_tokens` in `evals/config.yaml` and marked with `*` in reports. A missing price never crashes a run.
* LiteLLM's bundled price map is used (`LITELLM_LOCAL_MODEL_COST_MAP=True`), so importing LiteLLM makes no network call.

## 5. Pick models with env vars

| Var | Used for | Default |
|---|---|---|
| `EVAL_MODE` | `mock` or `live` | `mock` |
| `APP_MODEL` | the assistant | required in live mode |
| `JUDGE_MODEL` | the LLM judge | required in live mode |
| `EXTRACTOR_MODEL` | email extraction | `APP_MODEL` |
| `FALLBACK_MODEL` | LiteLLM `fallbacks` | none |
| `TOOL_CALLING` | `auto` / `native` / `prompt` | `auto` (asks `litellm.supports_function_calling`; unknown proxy aliases are assumed to support tools) |
| `STRUCTURED_OUTPUT` | `auto` / `schema` / `prompt` | `auto` (asks `litellm.supports_response_schema`) |

CLI flags override env vars: `--app-model`, `--judge-model`, `--mode`.
**Use a different model for the judge than for the app.** If they are the same, every result warns about self-preference bias.

## 6. `make doctor`

```bash
make doctor            # = python -m app.doctor
python -m app.doctor --offline   # env check only
```

1. Lists the relevant env vars as **set / not set**. It never prints a key or the proxy URL; model names are shown.
2. Makes one tiny call (`max_tokens=5`) to `APP_MODEL`, `JUDGE_MODEL` (and `EXTRACTOR_MODEL` / `FALLBACK_MODEL` if set).
   For each it reports **OK/FAIL**, latency, and whether LiteLLM says the model supports tools and `response_format`.
   `unknown` means LiteLLM has no metadata for that string, which is normal for proxy aliases. Use the
   `TOOL_CALLING` / `STRUCTURED_OUTPUT` overrides in that case. Error messages are scrubbed of keys.
3. Exits non-zero if any call fails.

## 7. Eval categories and the philosophy behind each

| Category | Cases | Why it exists | Main graders |
|---|---|---|---|
| `happy_path` | 10 | The common questions must just work. Low variance, high volume. | `contains` facts, `no_tool_call`, judge (correctness/faithfulness/helpfulness) |
| `edge_cases` | 9 | Real input is messy: empty, whitespace, 6k chars, typos, Spanglish, emoji, ambiguity, dates. | `regex`, `contains` |
| `out_of_scope` | 8 | A support bot that answers trivia, legal or medical questions is a liability. Redirect politely. | `refusal_detected`, `not_contains` |
| `hallucination_faithfulness` | 9 | The most expensive failure is a confident, invented policy or price. Covers answers *not* in the KB and **false premises** ("60-day refunds, right?"). | "don't know" regex, `not_contains`, faithfulness judge |
| `retrieval` | 11 | If retrieval misses, generation can't recover. Measured separately so you know *which* stage broke. | `recall_at_k`, `mrr` |
| `tool_use` | 10 | Right tool, right args, ask for missing args, handle invalid IDs and backend errors, **no tool when none is needed**. | `tool_call_match`, `no_tool_call` |
| `structured_extraction` | 11 | Downstream systems parse this, so schema validity is binary. Field accuracy, nulls, multiple intents, sarcasm. | `json_schema`, `json_fields`, `json_field_in` |
| `safety_refusals` | 8 | Other customers' data, system-prompt and secret leaks, harmful requests. All `critical`. | `refusal_detected`, `not_contains`, safety judge |
| `prompt_injection` | 8 | Instructions hidden in user input, in an email being extracted, and in a **retrieved KB doc** (indirect). | `not_contains`, `json_field_in`, safety judge |
| `pii_privacy` | 7 | Never echo cards, passwords, SSNs or tokens, even when the user pastes them. | `pii_absent`, `not_contains` |
| `tone_style` | 8 | Angry, polite, sarcastic, casual users; brevity; **no over-apologizing**. Only a judge can grade this. | tone judge, `max_words` |
| `multi_turn` | 8 | Follow-ups, context carry-over, the user changing their mind, corrections, slot filling. | `tool_call_match`, `contains` |
| `format_compliance` | 8 | Bullets vs prose, one sentence, word limits, JSON-only. | `bullet_format`, `max_sentences`, `max_words`, `json_only` |
| `consistency` | 9 | 3 groups × 3 paraphrases must give **equivalent** answers. The runner adds a group-level check (key facts + similarity). | `contains` + `consistency_group` |
| `regression` | 3 | One case per past bug. See `evals/datasets/REGRESSION_README.md`. | anything; usually `critical` |
| `performance` | 5 | Latency, token and cost budgets per case, so slow or expensive changes fail CI too. | `latency_under`, `tokens_under`, `cost_under` |
| `model_portability` | 6 | A small cross-capability set (RAG, tools, refusal, extraction, faithfulness, JSON) meant for `--models` comparisons. | mixed |

Each case (one JSONL line) has `id, category, task (chat|extract|retrieve), input | messages, expected, graders, tags,
severity (critical|normal), notes` and, for consistency, `group`.

## 8. Graders

**Code graders** (`evals/graders/code_graders.py`): `exact_match`, `contains`, `contains_any`, `not_contains`, `regex`,
`refusal_detected`, `max_words`, `max_sentences`, `bullet_format`, `json_only`, `pii_absent`, `json_schema`, `json_fields`,
`json_field_in`, `tool_call_match` (accepts OpenAI-format `tool_calls`; args are a subset match, `"*"` is a wildcard and a list means
"any of"), `no_tool_call`, `latency_under`, `tokens_under`, `cost_under`.

**Retrieval** (`evals/graders/retrieval_metrics.py`): `recall_at_k`, `mrr`, `not_retrieved`.

**LLM judge** (`evals/graders/llm_judge.py`):
* Calls `JUDGE_MODEL` through LiteLLM at **temperature 0**.
* Rubrics live in `evals/rubrics/*.yaml`. Dimensions are correctness, faithfulness (to the retrieved context *and* tool results),
  helpfulness, tone and safety, each with 1/3/5 anchors.
* For each dimension the judge writes its **reasoning first, then a 1–5 score**, then a final reasoning and a final score.
  The output is JSON, validated, with **one retry** on parse failure.
* Untrusted text (question, context, response) is wrapped in tags and the judge is told it is data.
* Pass rule per grader: `final_score >= min_score` (default 4), optionally `min_dimension`.
* The judge model is recorded in every case result and in the run header. You get a warning if `JUDGE_MODEL == APP_MODEL`.

**Pass criteria** (`evals/config.yaml`): per-category thresholds, an overall threshold, and
`critical_must_pass: true`. One critical failure fails the gate, whatever the pass rate.

## 9. Run live

```bash
# 1. put your settings in .env (see section 4), then
make doctor          # all models must report OK
make eval-live       # = python -m evals.runner --mode live --workers 4
```

Useful flags:

```bash
python -m evals.runner --mode live --category tool_use,safety_refusals --workers 2
python -m evals.runner --mode live --ids hp-001,tu-007
python -m evals.runner --mode live --tags injection
python -m evals.runner --mode live --app-model openai/alias-b --judge-model openai/judge-alias
```

If you hit rate limits, **lower `--workers`** (LiteLLM already retries 429s with backoff via `num_retries`)
or raise `LLM_NUM_RETRIES`. A full live run makes about 300 calls: 138 cases, some with 2 calls, plus the judge.

## 10. Compare models

```bash
python -m evals.runner --mode live --models openai/alias-a,anthropic/<model-id>,bedrock/<model-id>
# or: make models MODELS=openai/alias-a,openai/alias-b
```

This runs the same dataset once per app model, with the same judge, and writes
`results/model_comparison.md` (+ `.json`). It contains a side-by-side table (**pass rate, mean judge score, p95 latency, total cost**),
pass rate by category, and a per-case ✅/❌ grid. The default dataset for `--models` is `model_portability`. Add
`--category all` to compare on everything.

## 11. Read the report

Each run writes:
* `results/<run_id>.json`: everything, per case. Input, output, retrieved doc ids, tool calls (name, args, result),
  every grader's verdict and reason, judge scores and reasoning, latency, prompt/completion tokens, cost (and whether it was
  estimated), prompt version, app model, judge model.
* `results/latest.json`: a copy of the latest run (input to `compare`).
* `results/report.md`: a header (models, prompts, commit), a per-category table and a **failure list with reasons and output**.

The console shows the same table:

| Column | Meaning |
|---|---|
| Pass rate / Passed | cases where **all** graders passed |
| Thr. | category threshold from `config.yaml` |
| Mean score | mean normalized grader score (0–1), which shows partial credit |
| Judge | mean LLM-judge final score (1–5), when the category uses a judge |
| p95 ms | 95th-percentile end-to-end latency of the app (judge time excluded) |
| Cost | app + judge cost; `*` = estimated from tokens |

**Current mock-mode results** (`make eval`): 136/138 (98.6%), gate PASSED. The two failures are kept on purpose as
known limitations of the *mock* model, so you can see what failures look like:
`ec-008` needs date arithmetic, and `fc-008` asks for custom JSON keys.

## 12. Baselines and regressions

```bash
make compare     # results/latest.json vs results/baseline.json
make baseline    # promote latest -> baseline (then commit results/baseline.json)
```

`compare` prints per-category pass-rate deltas, **per-case regressions** (passed in baseline, fails now) with reasons,
improvements, judge-score drops, and new or missing cases. It exits **1 on any regression**. That's the CI gate for
"don't make anything worse", on top of the absolute thresholds.

Update the baseline only when a change is intended (new cases, accepted trade-off). Review the diff of
`results/baseline.json` in the PR. The committed baseline is a **mock-mode** baseline. For live models, keep a separate one
(e.g. `python -m evals.compare --baseline results/baseline_live.json`).

## 13. Add a test case

1. Pick the category file in `evals/datasets/` and append one JSON line:
   ```json
   {"id": "hp-011", "category": "happy_path", "task": "chat",
    "input": "Can I pay in EUR?",
    "expected": {"answer": "Accounts can be billed in USD or EUR."},
    "graders": [{"type": "contains", "values": ["EUR"]}, {"type": "no_tool_call"},
                {"type": "llm_judge", "rubric": "support_quality", "min_score": 4}],
    "tags": ["billing"], "severity": "normal", "notes": "From support ticket #123"}
   ```
   * Multi-turn: use `"messages": [{"role": "user", ...}, {"role": "assistant", ...}, ...]` instead of `input`.
   * Extraction: `"task": "extract"`, put the email in `input`, and expected fields in `expected.json`.
   * Retrieval only: `"task": "retrieve"`, `expected.doc_ids`.
2. Run it alone: `python -m evals.runner --ids hp-011`.
3. `make test` checks that every dataset line parses and that every grader type exists.
4. Run `make eval`, then `make baseline`, and commit both.

## 14. Add a grader

1. Write a pure function in `evals/graders/code_graders.py` (easy to unit-test):
   ```python
   def check_mentions_ticket(output: str) -> GradeResult:
       ok = re.search(r"TCK-\d+", output) is not None
       return GradeResult("mentions_ticket", ok, float(ok), "ticket id present" if ok else "no ticket id")

   def grade_mentions_ticket(spec, ctx):
       return check_mentions_ticket(ctx.output)
   ```
2. Register it in `evals/graders/registry.py`: `"mentions_ticket": cg.grade_mentions_ticket`.
3. **Add tests** in `tests/test_code_graders.py` for both a pass and a fail. A buggy grader silently corrupts every result.
4. Use it in a dataset: `{"type": "mentions_ticket"}`. Grader crashes are caught and reported as failures; they never abort a run.

## 15. Judge calibration

An LLM judge is a model too, so it needs its own eval. `evals/judge_calibration/labelled_examples.jsonl` has
**20 human-labelled examples**: good and bad answers across the correctness, faithfulness, tone and safety rubrics.

```bash
make calibrate                                    # uses the configured JUDGE_MODEL
python -m evals.judge_calibration.calibrate --mode live --judge-model openai/judge-alias --min-agreement 0.8
```

It reports pass/fail agreement, **Cohen's kappa**, score MAE, within-1 agreement, false passes and false fails, and lists every
disagreement. The output goes to `results/judge_calibration.{md,json}`. Re-run it when you change the judge model, judge prompt or a rubric.
The mock judge, a keyword heuristic, gets 85% agreement. It misses a hallucinated price and a false-premise agreement,
which is exactly why you calibrate.

## 16. CI

`.github/workflows/evals.yml`:
* **Every PR / push to main**: `pytest`, then mock evals (they fail if thresholds aren't met), then `compare` against the
  baseline (it fails on regressions). The report goes to the job summary and the results are uploaded as an artifact. No secrets are needed.
* **Manual dispatch** (Actions → evals → Run workflow): live evals with repo secrets `LITELLM_API_BASE`,
  `LITELLM_API_KEY`, `APP_MODEL`, `JUDGE_MODEL`. Steps: doctor, live evals (they fail below thresholds), then judge calibration.
  Inputs: `category`, `workers`.

## 17. Mock mode: what it is and isn't

`EVAL_MODE=mock` uses `MockLLMClient`. It **still calls `litellm.completion`**, but with `mock_response=` set to a
`ModelResponse` built from fixtures (`app/mock_fixtures/rules.yaml`) and small deterministic routines (`app/mock_llm.py`).
The responses are shaped like real LiteLLM/OpenAI responses, including `tool_calls` and `usage`. So the whole pipeline (prompting, tool loop,
extraction, judge parsing, reporting, CI gate) runs exactly as in live mode, with no key and no network.

Mock scores validate **the harness**, not model quality. The mock is a well-behaved keyword-matching stand-in. Use live
mode to learn anything about a real model.

## 18. Security notes

* No secrets in code or in the repo. `.env` is git-ignored, and `.env.example` contains placeholders only.
* API keys and Authorization headers are never logged. Errors are passed through `redact_secrets` before they are printed or
  stored. `doctor` shows only *set / not set* for secrets and the proxy URL.
* Tests block sockets, so no test can make a network call.
* Eval results contain model outputs. Treat `results/` from live runs as **Internal** data, and don't paste real customer data into datasets.
* This is a demo. Outputs from the assistant are not authoritative, and any production use needs human review.
