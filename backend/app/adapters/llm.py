"""AI risk investigator.

The language model rewrites a deterministic summary more fluently. That is all
it does. It does not score, does not decide, and never sees a policy action it
could contradict, because it is not given one.

Three defences, in order of how much they matter:

1.  THE MODEL IS NOT IN THE DECISION PATH. Detection, scoring and policy all
    complete before this module is called. Its output is displayed and logged;
    nothing reads it back.

2.  THE PROMPT CARRIES ONLY STRUCTURED EVIDENCE. No raw transactions, no
    customer identifiers, no recommended action. The model cannot leak what it
    was never shown, and cannot agree or disagree with a verdict it never saw.

3.  OUTPUT IS VALIDATED BEFORE IT IS SHOWN. If the model states a verdict,
    invents a figure that is not in the evidence, or returns something
    unusable, the deterministic text is used instead and the response is marked
    degraded.

The deterministic path is the DEFAULT, not a fallback of last resort. With no
API key configured the product works exactly as designed, which is why the
templated text is written to stand on its own.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

# Language a merchant might act on. An explanation stating any of these is
# claiming authority the model does not have, so it is rejected outright.
VERDICT_LANGUAGE = re.compile(
    r"\b(block(ed|ing)?|ban(ned|ning)?|suspend(ed)?|freeze|frozen|"
    r"reject(ed)?|deny|denied|refund them|charge ?back them|"
    r"you should (block|ban|suspend|reject)|"
    r"definitely fraud|certainly fraud|is fraud|are fraudsters|guilty)\b",
    re.IGNORECASE,
)

# Certainty the evidence does not support.
OVERCLAIM_LANGUAGE = re.compile(
    r"\b(proves?|proven|conclusively|without doubt|no doubt|"
    r"100% certain|undeniably)\b",
    re.IGNORECASE,
)

MAX_EXPLANATION_CHARS = 900

SYSTEM_PROMPT = (
    "You rewrite fraud-analysis findings for a merchant. You are given only "
    "structured evidence that has already been computed. Restate it in clear, "
    "plain prose.\n\n"
    "Rules you must follow:\n"
    "- Use only the numbers given. Never invent, round differently, or infer "
    "new figures.\n"
    "- Never recommend an action. You do not decide what happens.\n"
    "- Never assert certainty. These are probabilistic findings about a "
    "connected group of accounts, not proof about people.\n"
    "- Distinguish what was observed from what the model inferred.\n"
    "- Two or three sentences. No headings, no lists, no preamble."
)


@dataclass(frozen=True)
class Explanation:
    source: str
    text: str
    facts: list[str]
    inferences: list[str]
    degraded: bool

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "facts": self.facts,
            "inferences": self.inferences,
            "degraded": self.degraded,
        }


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def complete(self, system: str, user: str) -> str: ...


class NullProvider:
    """No credentials configured. The product's normal state."""

    name = "none"

    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("no language model configured")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5"):
        self._api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self._model = model

    def available(self) -> bool:
        return bool(self._api_key)

    def complete(self, system: str, user: str) -> str:
        import json
        import urllib.request

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self._model,
                    "max_tokens": 400,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
            ).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            body = json.loads(response.read())
        return "".join(
            block.get("text", "") for block in body.get("content", [])
        ).strip()


def get_provider() -> Provider:
    """Chosen from the environment. Absent credentials is not an error."""
    if os.environ.get("RAZORSHIELD_DISABLE_LLM") == "1":
        return NullProvider()
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic" and os.environ.get("LLM_API_KEY"):
        return AnthropicProvider()
    return NullProvider()


def build_prompt(ring: dict) -> str:
    """Structured evidence only.

    Note what is absent: no customer ids, no transaction rows, no addresses, and
    critically no recommended action. The model is asked to explain findings,
    not to endorse a decision.
    """
    summary = ring["summary"]
    lines = [
        "FINDINGS FOR ONE CONNECTED GROUP OF ACCOUNTS",
        f"- accounts: {summary['n_accounts']}",
        f"- shared devices: {summary['n_devices']}",
        f"- shared delivery addresses: {summary['n_addresses']}",
        f"- shared IP addresses: {summary['n_ips']}",
        f"- orders: {summary['n_transactions']}",
        f"- model risk score: {summary['risk_score']:.0f} out of 100",
        f"- money at stake: INR {summary['financial_exposure']:,.0f}",
        "",
        "EVIDENCE, ordered by how much each signal contributed:",
    ]
    for item in ring["evidence"]:
        share = f" [{item['weight']:.0%} of the decision]" if item["weight"] else ""
        lines.append(f"- ({item['kind'].lower()}) {item['statement']}{share}")
    return "\n".join(lines)


def validate(text: str, ring: dict) -> tuple[bool, str]:
    """Reject anything that oversteps. Returns (ok, reason)."""
    if not text or len(text.strip()) < 40:
        return False, "response was empty or too short to be useful"
    if len(text) > MAX_EXPLANATION_CHARS:
        return False, "response exceeded the length limit"

    match = VERDICT_LANGUAGE.search(text)
    if match:
        return False, f"response recommended an action ('{match.group(0)}')"

    match = OVERCLAIM_LANGUAGE.search(text)
    if match:
        return False, f"response asserted certainty ('{match.group(0)}')"

    # Figures must come from the evidence. A model that invents a plausible
    # rupee number is more dangerous than one that says nothing.
    allowed = {
        str(int(ring["summary"]["financial_exposure"])),
        f"{int(ring['summary']['financial_exposure']):,}",
        str(int(ring["summary"]["risk_score"])),
        str(ring["summary"]["n_accounts"]),
        str(ring["summary"]["n_devices"]),
        str(ring["summary"]["n_addresses"]),
        str(ring["summary"]["n_ips"]),
        str(ring["summary"]["n_transactions"]),
        "100",
    }
    for item in ring["evidence"]:
        value = item["observed_value"]
        allowed.update({str(value), str(int(float(value)))} if _numeric(value) else {})
        if item["baseline_value"] is not None and _numeric(item["baseline_value"]):
            allowed.add(str(int(float(item["baseline_value"]))))

    for number in re.findall(r"\d[\d,]*\.?\d*", text):
        cleaned = number.rstrip(".")
        if cleaned in allowed:
            continue
        stripped = cleaned.replace(",", "")
        if stripped in allowed:
            continue
        try:
            if str(int(float(stripped))) in allowed:
                continue
        except ValueError:
            pass
        return False, f"response contained a figure not in the evidence ({cleaned})"

    return True, ""


def _numeric(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def explain(
    ring: dict,
    deterministic_text: str,
    provider: Provider | None = None,
) -> tuple[Explanation, str]:
    """Return an explanation and a log line describing how it was produced."""
    facts = [e["statement"] for e in ring["evidence"] if e["kind"] == "FACT"]
    inferences = [e["statement"] for e in ring["evidence"] if e["kind"] == "INFERENCE"]

    def fallback(reason: str) -> tuple[Explanation, str]:
        return (
            Explanation(
                source="deterministic",
                text=deterministic_text,
                facts=facts,
                inferences=inferences or [deterministic_text],
                degraded=True,
            ),
            f"AI explanation service unavailable — deterministic fallback used: {reason}",
        )

    provider = provider or get_provider()
    if not provider.available():
        return (
            Explanation(
                source="deterministic",
                text=deterministic_text,
                facts=facts,
                inferences=inferences or [deterministic_text],
                # Not degraded: no model is configured, which is a supported
                # configuration rather than a failure.
                degraded=False,
            ),
            "No language model configured; deterministic explanation used.",
        )

    try:
        text = provider.complete(SYSTEM_PROMPT, build_prompt(ring))
    except Exception as exc:  # noqa: BLE001 - any provider failure degrades
        return fallback(f"{type(exc).__name__}")

    ok, reason = validate(text, ring)
    if not ok:
        return fallback(reason)

    return (
        Explanation(
            source="llm",
            text=text.strip(),
            facts=facts,
            inferences=inferences or [deterministic_text],
            degraded=False,
        ),
        "Language model explanation generated and validated.",
    )
