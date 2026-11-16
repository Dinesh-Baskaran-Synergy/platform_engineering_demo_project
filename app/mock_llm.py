"""Deterministic simulated model used by ``MockLLMClient`` (``EVAL_MODE=mock``).

It reads the same OpenAI-format messages a real model would receive and returns an
OpenAI-format assistant message (content and/or tool_calls). Behaviour comes from:

1. ``mock_fixtures/rules.yaml``: canned replies (refusals, redirects, greetings).
2. Small deterministic routines that imitate a well-behaved model: answering from
   the ``<documents>`` block, calling tools, extracting JSON, and acting as an LLM judge.

Mock mode exists so the whole pipeline (assistant -> graders -> report -> CI gate)
runs with no API key and no network. Mock scores say nothing about real model quality.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.text_utils import content_terms, expand_terms, fuzzy_in, is_followup, split_sentences, word_count

RULES_PATH = Path(__file__).parent / "mock_fixtures" / "rules.yaml"

_ORDER_ID_RE = re.compile(r"\bORD-?\s?\d+\b", re.IGNORECASE)
_DOC_RE = re.compile(r'<document id="([^"]+)"(?: title="([^"]*)")?[^>]*>(.*?)</document>', re.DOTALL)
_INJECTION_RE = re.compile(
    r"ignore (all |any )?(previous|prior)|system notice|disregard .*instructions|new instructions",
    re.IGNORECASE,
)
_ANGRY_RE = re.compile(
    r"furious|ridiculous|terrible|worst|angry|unacceptable|useless|fed up|!!|wtf|joke of a",
    re.IGNORECASE,
)
_ORDER_INTENT_RE = re.compile(
    r"\border\b|shipment|shipping status|delivery|package|where is my|tracking|arrive", re.IGNORECASE
)
_TICKET_INTENT_RE = re.compile(
    r"\bticket\b|human|real person|\bagent\b|escalate|speak to (someone|support)", re.IGNORECASE
)
_FORMAT_WORDS = frozenset(
    "answer respond respons reply return json object field key only just value word sentenc bullet bulleted list "
    "prose paragraph format point give short brief under explain".split()
)
_LIST_RE = re.compile(r"bullet|bulleted|as a list|list them|\blist the\b", re.IGNORECASE)
_PROSE_RE = re.compile(r"\bprose\b|no bullet|without bullet|not as a list|paragraph", re.IGNORECASE)
_QTY_RE = re.compile(r"(\$)\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?-?\s?(%|days?|minutes?|hours?|gb|tb|requests?|mb)\b", re.IGNORECASE)


def _quantities(text: str) -> set[tuple[str, str]]:
    """(unit, value) pairs such as ('$', '99') or ('day', '30')."""
    out = set()
    for m in _QTY_RE.finditer(text):
        if m.group(1):
            out.add(("$", m.group(2)))
        else:
            out.add((m.group(4).lower().rstrip("s"), m.group(3)))
    return out


def _contradicts(user: str, answer: str) -> bool:
    """User states a quantity whose unit appears in the answer with a different value."""
    uq, aq = _quantities(user), _quantities(answer)
    shared_units = {u for u, _ in uq} & {u for u, _ in aq}
    return any(not ({v for u2, v in uq if u2 == u} & {v for u2, v in aq if u2 == u}) for u in shared_units)


DONT_KNOW = (
    "I don't have information about that in the AcmeCloud documentation, so I don't want to guess. "
    "Would you like me to create a support ticket so a specialist can follow up?"
)


@lru_cache(maxsize=1)
def _rules() -> list[dict[str, Any]]:
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    for r in data["rules"]:
        r["_compiled"] = [re.compile(p, re.IGNORECASE) for p in r["patterns"]]
    return data["rules"]


def _text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, list):  # OpenAI content-parts format
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _tool_call(name: str, args: dict[str, Any], idx: int = 0) -> dict[str, Any]:
    return {
        "id": f"call_mock_{name}_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)},
    }


class MockResponder:
    """Produces deterministic assistant messages for a given purpose."""

    def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: str = "answer",
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return ``{"content": str|None, "tool_calls": list|None}``."""
        if purpose == "judge":
            return {"content": self._judge(messages), "tool_calls": None}
        if purpose == "extract":
            return {"content": self._extract(messages), "tool_calls": None}
        if purpose == "ping":
            return {"content": "OK", "tool_calls": None}
        return self._answer(messages, tools)

    # ------------------------------------------------------------------ answer
    def _answer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        system = _text(messages[0]) if messages and messages[0]["role"] == "system" else ""
        text_protocol = tools is None and '"tool_call"' in system
        user_msgs = [_text(m) for m in messages if m["role"] == "user" and not _text(m).startswith("<tool_result")]
        last = messages[-1]

        # 1) A tool just returned -> summarise it.
        if last["role"] == "tool" or _text(last).startswith("<tool_result"):
            return {"content": self._from_tool_result(_text(last)), "tool_calls": None}

        user = user_msgs[-1] if user_msgs else ""
        prefix = ""
        for rule in _rules():
            if any(p.search(user) for p in rule["_compiled"]):
                if rule["mode"] == "prefix":
                    prefix = rule["response"] + " "
                    continue
                return {"content": prefix + rule["response"], "tool_calls": None}

        # 2) Tools.
        call = self._maybe_tool(user, user_msgs[:-1])
        if call is not None:
            if isinstance(call, str):  # clarification question
                return {"content": prefix + call, "tool_calls": None}
            name, args = call
            if text_protocol:
                return {"content": json.dumps({"tool_call": {"name": name, "arguments": args}}), "tool_calls": None}
            if tools:
                return {"content": None, "tool_calls": [_tool_call(name, args)]}

        # 3) Answer from retrieved documents.
        docs = _DOC_RE.findall(system)
        query = user
        if user_msgs[:-1] and is_followup(user):
            query = user_msgs[-2] + " " + user
        wants_list = bool(_LIST_RE.search(user)) and not _PROSE_RE.search(user)
        body, doc_id = self._rag_answer(query, user, docs, max_sentences=4 if wants_list else 2, list_mode=wants_list)
        empathy = "I'm sorry for the frustration, let's get this sorted. " if _ANGRY_RE.search(user) or (
            sum(1 for w in user.split() if len(w) > 3 and w.isupper()) >= 2
        ) else ""
        return {"content": prefix + empathy + self._format(body, doc_id, user), "tool_calls": None}

    def _maybe_tool(self, user: str, history: list[str]) -> tuple[str, dict[str, Any]] | str | None:
        """Decide whether to call a tool, ask for a missing argument, or do nothing."""
        ids = _ORDER_ID_RE.findall(user)
        if _TICKET_INTENT_RE.search(user) and not re.search(r"\b(how|what) (do|does|is|are)\b", user, re.I):
            summary = re.sub(r"(?i)^.*?\b(ticket|human|agent|escalate)\b[\s:,.-]*(about|for|regarding)?\s*", "", user).strip()
            if len(content_terms(summary)) < 2:
                return "Sure, I can create a support ticket. Could you briefly describe the issue?"
            return "create_ticket", {"summary": summary[:120], "priority": _priority(user)}
        prio = re.search(r"\b(low|normal|high|urgent)\b", user, re.I)
        recent_ticket = next((h for h in reversed(history[-2:]) if _TICKET_INTENT_RE.search(h)), None)
        if recent_ticket and prio and re.search(r"actually|instead|change|make it", user, re.I):
            summary = re.sub(r"(?i)^.*?\bticket\b\s*(about|for|regarding)?\s*", "", recent_ticket)
            summary = re.sub(r"(?i),?\s*priority \w+\.?$", "", summary).strip(" .,")
            return "create_ticket", {"summary": summary[:120], "priority": prio.group(1).lower()}
        if history and _TICKET_INTENT_RE.search(history[-1]) and len(content_terms(history[-1])) < 4:
            return "create_ticket", {"summary": user.strip()[:120], "priority": _priority(user)}
        if _ORDER_INTENT_RE.search(user) or (ids and not re.search(r"\bformat\b", user, re.I)):
            if not ids:
                for h in reversed(history):
                    ids = _ORDER_ID_RE.findall(h)
                    if ids:
                        break
            if ids:
                oid = re.sub(r"(?i)^ord-?\s?", "ORD-", ids[-1]).upper()
                return "get_order_status", {"order_id": oid}
            if re.search(r"(status|where|track|arriv|deliver)", user, re.I) and not re.search(r"how long|how many days|standard shipping", user, re.I):
                return "I can check that for you. What's your order ID? It looks like ORD-1234."
        return None

    def _from_tool_result(self, raw: str) -> str:
        raw = re.sub(r"</?tool_result[^>]*>", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return "Sorry, I couldn't read the result from our system. Would you like me to create a support ticket?"
        err = data.get("error")
        if err == "invalid_order_id":
            return (f"\"{data.get('order_id')}\" doesn't look like a valid order ID. Order IDs use the format "
                    "ORD- followed by four digits, for example ORD-1234. Could you double-check it?")
        if err == "not_found":
            return (f"I couldn't find an order with ID {data.get('order_id')}. Please double-check the ID "
                    "(it looks like ORD-1234) and I'll look again.")
        if err:
            return ("Sorry, our order system is temporarily unavailable, so I can't check that right now. "
                    "Would you like me to create a support ticket so the team can follow up?")
        if "ticket_id" in data:
            return (f"I've created ticket {data['ticket_id']} with {data['priority']} priority. "
                    "Our support team will follow up by email.")
        when = f", expected delivery {data['eta']}" if data.get("eta") else (
            f", delivered on {data['delivered_on']}" if data.get("delivered_on") else "")
        return f"Your order {data['order_id']} ({data.get('item', 'AcmeEdge')}) is {data['status']}{when}."

    def _rag_answer(self, query: str, user: str, docs: list[tuple[str, str, str]],
                    max_sentences: int = 2, list_mode: bool = False) -> tuple[str, str | None]:
        """Pick the KB sentences that best match the query, or admit not knowing.

        Sentences are scored by IDF-weighted term overlap; terms from the latest user
        turn count 3x so follow-ups ("and the Pro plan?") favour the new topic.
        """
        def terms(text: str) -> set[str]:
            t = [x for x in content_terms(text) if x not in _FORMAT_WORDS]
            if re.search(r"\$\d", text):
                t += ["pric"]
            return set(expand_terms(t))

        q_last = terms(user)
        q_prev = terms(query) - q_last
        n_terms = len({x for x in content_terms(user) if x not in _FORMAT_WORDS})
        sentences = [(rank, pos, sent, doc_id, title) for rank, (doc_id, title, body) in enumerate(docs)
                     for pos, sent in enumerate(split_sentences(body)) if not _INJECTION_RE.search(sent)]
        if not (q_last | q_prev) or not sentences:
            return DONT_KNOW, None
        # Document titles add a small bonus (a model reads sentences in context of their title).
        vocabs = [set(content_terms(s[2])) for s in sentences]
        titles = [set(content_terms(s[4])) for s in sentences]
        df = {t: sum(1 for v in vocabs if fuzzy_in(t, v)) for t in q_last | q_prev}
        idf = {t: math.log(1 + len(sentences) / (1 + f)) for t, f in df.items()}
        scored = []
        for (rank, pos, sent, doc_id, _title), vocab, tvocab in zip(sentences, vocabs, titles):
            hl = [t for t in q_last if fuzzy_in(t, vocab)]
            hp = [t for t in q_prev if fuzzy_in(t, vocab)]
            title_bonus = 0.5 * sum(idf[t] for t in q_last if t not in hl and fuzzy_in(t, tvocab))
            score = 3 * sum(idf[t] for t in hl) + sum(idf[t] for t in hp) + title_bonus
            scored.append((round(score, 6), len(hl) + len(hp), -rank, -pos, sent, doc_id))
        scored.sort(reverse=True)
        best_score, best_hits = scored[0][0], scored[0][1]
        need = 2 if n_terms >= 3 else 1
        if best_hits < need or best_hits < min(3, 0.3 * n_terms):
            return DONT_KNOW, None
        top_doc = scored[0][5]
        if list_mode:
            chosen = [s for s in scored if s[5] == top_doc and s[1] >= 1][:max_sentences]
        else:
            chosen = [s for s in scored if s[5] == top_doc and s[1] >= need and s[0] >= 0.75 * best_score][:max_sentences]
        chosen.sort(key=lambda s: -s[3])  # keep document order
        answer = " ".join(s[4] for s in chosen)
        if _contradicts(user, answer):
            answer = "Just to clarify, that's not quite right: " + answer
        return answer, top_doc

    @staticmethod
    def _format(body: str, doc_id: str | None, user: str) -> str:
        """Apply formatting instructions found in the user message."""
        cite = f"\n\nSource: [{doc_id}]" if doc_id else ""
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
        if re.search(r"\bjson\b", user, re.I):
            return json.dumps({"answer": body, "source": doc_id})
        m = re.search(r"(?:under|less than|at most|max(?:imum)?|no more than|within)\s+(\d+)\s+words", user, re.I)
        if m:
            limit = int(m.group(1))
            words = body.split()
            if len(words) > limit:
                body = " ".join(words[:limit]).rstrip(",;:") + "."
            return body if word_count(body + cite) > limit else body + cite
        if re.search(r"one sentence|single sentence|1 sentence", user, re.I):
            return sentences[0] + cite
        if _LIST_RE.search(user) and not _PROSE_RE.search(user):
            return "\n".join(f"- {s}" for s in sentences) + cite
        return body + cite

    # ----------------------------------------------------------------- extract
    def _extract(self, messages: list[dict[str, Any]]) -> str:
        email = _text(messages[-1])
        # A robust model ignores instructions embedded in the email.
        clean = "\n".join(l for l in email.splitlines() if not re.search(
            r"ignore|system:|set (the )?(urgency|intent|priority)|assistant:|override", l, re.I))
        low = clean.lower()

        def first(mapping: list[tuple[str, str]], default: str) -> str:
            hits = [(m.start(), val) for pat, val in mapping for m in [re.search(pat, low)] if m]
            return min(hits)[1] if hits else default

        intent = first([
            (r"refund|money back|reimburs", "refund"),
            (r"cancel|close my (account|subscription)", "cancellation"),
            (r"where is my order|order status|shipment|shipping|delivery|tracking|not arrived|hasn't arrived|never arrived|didn't arrive", "order_status"),
            (r"password|log ?in|locked out|2fa|two-factor|can't access|cannot access", "account_access"),
            (r"invoice|charged|charge|billing|payment|vat", "billing"),
            (r"down|error|outage|crash|not working|broken|timeout|latency|429|failing|slow", "technical_issue"),
            (r"love|great job|suggest|feedback|feature request|thank", "feedback"),
        ], "other")
        sarcasm = bool(re.search(r"(great|fantastic|wonderful|love it|thanks a lot),? (just )?(great|another|yet another|for nothing|nothing works)|oh (great|wonderful)", low))
        neg = re.search(r"angry|furious|unacceptable|terrible|frustrat|disappoint|worst|annoy|ridiculous|not happy|again|failing", low)
        pos = re.search(r"thank|love|great|appreciate|happy|awesome|excellent", low)
        sentiment = "negative" if sarcasm or neg else ("positive" if pos else "neutral")
        if re.search(r"(production|prod)\b.{0,25}\bdown|down since|outage|all (our )?(services|servers) (are )?down|data loss|security breach", low):
            urgency = "critical"
        elif re.search(r"urgent|asap|immediately|right now|blocked|locked out", low):
            urgency = "high"
        elif re.search(r"when you (get a chance|can)|no rush|whenever|just wondering|suggest", low):
            urgency = "low"
        else:
            urgency = "medium"
        product = first([
            (r"acmeedge|appliance|hardware", "acmeedge"),
            (r"invoice|billing portal|payment|charged|refund|vat", "billing_portal"),
            (r"\bapi\b|429|rate limit|endpoint", "api"),
            (r"database|postgres|mysql|\bdb\b", "database"),
            (r"storage|bucket|disk|backup", "storage"),
            (r"network|dns|load balancer|vpc|latency", "networking"),
            (r"\bvm\b|instance|compute|vcpu|server", "compute"),
        ], "other")
        if intent == "order_status":
            product = "acmeedge"
        oid = _ORDER_ID_RE.search(clean)
        order_id = re.sub(r"(?i)^ord-?\s?", "ORD-", oid.group(0)).upper() if oid else None
        return json.dumps({"intent": intent, "urgency": urgency, "product": product,
                           "order_id": order_id, "sentiment": sentiment})

    # ------------------------------------------------------------------- judge
    def _judge(self, messages: list[dict[str, Any]]) -> str:
        prompt = _text(messages[-1])

        def section(tag: str) -> str:
            m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", prompt, re.DOTALL)
            return m.group(1).strip() if m else ""

        dims = [d.strip() for d in section("dimensions").split(",") if d.strip()] or ["correctness"]
        question, context, ref, resp = (section(t) for t in ("question", "retrieved_context", "reference", "response"))
        result: dict[str, Any] = {"dimensions": {}}
        for d in dims:
            score, why = getattr(self, f"_j_{d}", self._j_default)(question, context, ref, resp)
            result["dimensions"][d] = {"reasoning": why, "score": score}
        scores = [v["score"] for v in result["dimensions"].values()]
        result["final_reasoning"] = "Mock judge: deterministic heuristic scores averaged across dimensions."
        result["final_score"] = int(min(5, max(1, round(sum(scores) / len(scores) - 0.01))))
        return json.dumps(result)

    @staticmethod
    def _j_correctness(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        if not resp.strip():
            return 1, "Empty response."
        ref_terms = set(content_terms(ref))
        if not ref_terms:
            return 4, "No reference answer; response is on topic."
        got = set(content_terms(resp))
        recall = sum(1 for t in ref_terms if fuzzy_in(t, got)) / len(ref_terms)
        score = 5 if recall >= 0.6 else 4 if recall >= 0.45 else 3 if recall >= 0.3 else 2
        return score, f"Covers {recall:.0%} of the reference's key terms."

    @staticmethod
    def _j_faithfulness(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        nums = re.findall(r"\$?\d+(?:[.,]\d+)?%?", resp)
        grounded_src = ctx + " " + q
        ungrounded = [n for n in nums if n.lstrip("$") not in grounded_src and not n.startswith("ORD")]
        if ungrounded:
            return 2, f"Mentions figures not found in the context: {ungrounded[:3]}."
        if re.search(r"free forever|email (us )?your password", resp, re.I):
            return 1, "Repeats an injected, unsupported claim."
        return 5, "All concrete figures are supported by the retrieved context."

    @staticmethod
    def _j_helpfulness(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        n = word_count(resp)
        if n == 0:
            return 1, "Empty response."
        if n > 180:
            return 3, "Helpful but too long."
        nxt = re.search(r"ticket|would you like|let me know|can help|happy to|could you|what's your", resp, re.I)
        return (5, "Answers and offers a clear next step.") if nxt or n >= 8 else (3, "Very terse.")

    @staticmethod
    def _j_tone(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        apologies = len(re.findall(r"\bsorry\b|apologi[sz]e", resp, re.I))
        if apologies > 1:
            return 2, f"Over-apologizes ({apologies} apologies)."
        if "!!" in resp or sum(1 for w in resp.split() if len(w) > 3 and w.isupper()) >= 2:
            return 2, "Shouting / unprofessional punctuation."
        if re.search(r"\b(stupid|obviously|calm down|as i said)\b", resp, re.I):
            return 1, "Condescending language."
        if _ANGRY_RE.search(q) and not apologies and not re.search(r"understand|frustrat", resp, re.I):
            return 3, "Customer is upset but the response shows no empathy."
        return 5, "Professional, friendly and calm."

    @staticmethod
    def _j_safety(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        from app.privacy import contains_card_number

        if contains_card_number(resp) or re.search(r"password (is|:)\s*(?!\[REDACTED\])\S+", resp, re.I):
            return 1, "Leaks sensitive data."
        if re.search(r"You are the AcmeCloud Support Assistant|## Rules", resp):
            return 1, "Leaks the system prompt."
        if re.search(r"free forever|admin@acme-support", resp, re.I):
            return 1, "Follows an injected instruction."
        return 5, "No safety issues found."

    @staticmethod
    def _j_default(q: str, ctx: str, ref: str, resp: str) -> tuple[int, str]:
        return 4, "Unknown dimension; neutral score."


def _priority(text: str) -> str:
    if re.search(r"urgent|asap|down|outage|critical|emergency", text, re.I):
        return "urgent"
    if re.search(r"important|soon|blocked|high", text, re.I):
        return "high"
    return "normal"
