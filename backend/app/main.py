"""RazorShield API.

The service is deliberately a thin read layer over offline artifacts: generation,
feature building, model training, graph detection and simulation all happen
outside the request path. If metrics are absent, the API serves visibly marked
placeholder shapes; malformed or placeholder-labeled artifacts are hard errors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from . import fixtures
from .adapters import llm, razorpay
from .core import audit
from .core import financial as financial_engine
from .core import policy as policy_engine
from .core import recovery as recovery_engine
from .core.seed import seed as seed_audit
from .schemas import (
    AuditListResponse,
    Explanation,
    ChainVerification,
    DecisionRequest,
    MerchantDecision,
    FeedResponse,
    FinancialImpact,
    HealthResponse,
    MetricsResponse,
    PolicyConfigResponse,
    RecomputeRequest,
    RecoveryWorkflow,
    RecoveryWorkflowListResponse,
    RingDetail,
    RingListResponse,
    RingSummary,
    SimulationStartResponse,
    SimulationStep,
    SpikeListResponse,
)

VERSION = "0.1.0"
PHASE = 16

ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", Path(__file__).resolve().parents[2] / "artifacts")
)
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SPIKES_PATH = ARTIFACTS_DIR / "spikes.json"
RINGS_PATH = ARTIFACTS_DIR / "rings.json"
SIMULATION_PATH = ARTIFACTS_DIR / "simulation.json"

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

app = FastAPI(
    title="RazorShield API",
    version=VERSION,
    description="AI Merchant Risk Intelligence. Defensive analysis only.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _artifacts_present() -> bool:
    return METRICS_PATH.is_file()


def _llm_available() -> bool:
    """Presence of a key only. Never log or return the value itself."""
    return bool(os.environ.get("LLM_API_KEY"))


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=VERSION,
        artifacts_loaded=_artifacts_present(),
        llm_available=_llm_available(),
        phase=PHASE,
    )


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Serve real evaluation output when it exists, placeholders otherwise.

    A malformed artifact is a hard error, not a silent fallback to placeholders.
    Silently degrading here is exactly how a fake number reaches a demo.
    """
    if not _artifacts_present():
        return fixtures.placeholder_metrics()

    raw = json.loads(METRICS_PATH.read_text())
    try:
        parsed = MetricsResponse.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"artifacts/metrics.json violates the frozen contract. "
                f"Regenerate it with `make evaluate`. Details: {exc.error_count()} error(s)."
            ),
        ) from exc

    if parsed.is_placeholder:
        raise HTTPException(
            status_code=500,
            detail="artifacts/metrics.json is flagged is_placeholder=True. Refusing to serve it as real.",
        )
    return parsed


@app.get("/api/feed", response_model=FeedResponse)
def feed(limit: int = Query(default=50, ge=1, le=500)) -> FeedResponse:
    return fixtures.placeholder_feed(limit=limit)


def _load_rings() -> list[dict]:
    if not RINGS_PATH.is_file():
        return []
    return json.loads(RINGS_PATH.read_text())["rings"]


@app.get("/api/rings", response_model=RingListResponse)
def rings(limit: int = Query(default=50, ge=1, le=500)) -> RingListResponse:
    raw = _load_rings()
    if not raw:
        return fixtures.placeholder_rings()
    summaries = [RingSummary.model_validate(r["summary"]) for r in raw[:limit]]
    return RingListResponse(rings=summaries, total=len(raw))


@app.get("/api/rings/{ring_id}", response_model=RingDetail)
def ring_detail(ring_id: str) -> RingDetail:
    raw = _load_rings()
    if not raw:
        if not ring_id.startswith("AR-"):
            raise HTTPException(status_code=404, detail="Unknown ring id.")
        return fixtures.placeholder_ring_detail(ring_id)

    match = next((r for r in raw if r["ring_id"] == ring_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No ring with id {ring_id}.")
    try:
        return RingDetail.model_validate(
            {
                "summary": match["summary"],
                "evidence": match["evidence"],
                "graph": match["graph"],
                "policy": match["policy"],
                "explanation": match["explanation"],
                "component_features": match["component_features"],
            }
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "artifacts/rings.json violates the frozen contract. Regenerate "
                f"with `make rings`. {exc.error_count()} error(s)."
            ),
        ) from exc


@app.get("/api/recovery/workflows", response_model=RecoveryWorkflowListResponse)
def recovery_workflows(
    limit: int = Query(default=20, ge=1, le=100),
) -> RecoveryWorkflowListResponse:
    raw = _load_rings()
    workflows = recovery_engine.workflows_from_rings(raw, limit) if raw else []
    try:
        parsed = [RecoveryWorkflow.model_validate(w) for w in workflows]
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Recovery workflow generation violated the frozen contract. "
                f"{exc.error_count()} error(s)."
            ),
        ) from exc
    return RecoveryWorkflowListResponse(
        workflows=parsed,
        total=len(raw),
        caveat=(
            "Synthetic scenario workflow. No Razorpay production action is "
            "executed; every money-impacting step stops for merchant approval."
        ),
    )


@app.get("/api/rings/{ring_id}/explanation", response_model=Explanation)
def ring_explanation(
    ring_id: str,
    simulate_failure: bool = Query(default=False),
) -> Explanation:
    """Regenerate the investigator summary.

    Detection, scoring and policy have already completed by the time this runs.
    Whatever happens here, the case keeps its risk score and its recommended
    action -- a test asserts exactly that.
    """
    raw = _load_rings()
    match = next((r for r in raw if r["ring_id"] == ring_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No ring with id {ring_id}.")

    provider = llm.NullProvider() if simulate_failure else None
    if simulate_failure:
        class _Failing:
            name = "simulated-outage"

            def available(self) -> bool:
                return True

            def complete(self, system: str, user: str) -> str:
                raise ConnectionError("simulated language model outage")

        provider = _Failing()

    explanation, log_line = llm.explain(
        match, match["explanation"]["text"], provider=provider
    )
    audit.append(
        event="explanation_generated",
        actor="llm" if explanation.source == "llm" else "system",
        subject_id=ring_id,
        input_summary=f"{len(match['evidence'])} structured evidence items",
        decision=f"explanation source: {explanation.source}",
        reason=log_line,
        policy=match["policy"]["policy_version"],
        result="degraded" if explanation.degraded else "ok",
    )
    return Explanation.model_validate(explanation.to_dict())


@app.get("/api/spikes", response_model=SpikeListResponse)
def spikes() -> SpikeListResponse:
    if not SPIKES_PATH.is_file():
        return fixtures.placeholder_spikes()
    try:
        return SpikeListResponse.model_validate(json.loads(SPIKES_PATH.read_text()))
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "artifacts/spikes.json violates the frozen contract. "
                f"Regenerate with `make spikes`. {exc.error_count()} error(s)."
            ),
        ) from exc


@app.on_event("startup")
def _startup() -> None:
    razorpay.assert_test_mode()
    audit.init_db()
    seed_audit()


@app.get("/api/audit", response_model=AuditListResponse)
def audit_log(
    subject_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> AuditListResponse:
    rows = audit.entries(subject_id=subject_id, limit=limit)
    if not rows:
        return fixtures.placeholder_audit()
    return AuditListResponse(entries=rows, total=len(rows))


@app.get("/api/audit/verify", response_model=ChainVerification)
def audit_verify() -> ChainVerification:
    """Recompute the hash chain. Tamper evidence, not a health check."""
    return ChainVerification.model_validate(audit.verify_chain())


@app.post("/api/rings/{ring_id}/decision", response_model=MerchantDecision)
def record_decision(ring_id: str, req: DecisionRequest) -> MerchantDecision:
    """Record a merchant approval or dismissal.

    The decision is only ever appended to the audit log. There is no status
    column to update: a second, mutable source of truth could disagree with the
    trail, and then neither could be trusted.
    """
    raw = _load_rings()
    match = next((r for r in raw if r["ring_id"] == ring_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No ring with id {ring_id}.")

    policy = match["policy"]
    entry = audit.append(
        event="merchant_decision",
        actor="merchant",
        subject_id=ring_id,
        input_summary=(
            f"recommended {policy['recommended_action']} at risk "
            f"{match['summary']['risk_score']:.0f}"
        ),
        decision=req.decision,
        reason=req.note or "No note provided.",
        policy=policy["policy_version"],
        result="recorded",
    )
    return MerchantDecision(
        ring_id=ring_id,
        decision=req.decision,
        recorded_at=entry["timestamp"],
        audit_entry_id=entry["entry_id"],
    )


@app.get("/api/policy/config", response_model=PolicyConfigResponse)
def policy_config() -> PolicyConfigResponse:
    return PolicyConfigResponse.model_validate(policy_engine.config())


@app.post("/api/financial/recompute", response_model=FinancialImpact)
def financial_recompute(req: RecomputeRequest) -> FinancialImpact:
    """Re-run the impact sum over the stored held-out decisions.

    Every assumption is the merchant's to change. Nothing is scaled from a
    summary; the same decisions are re-totalled under the new numbers.
    """
    if not financial_engine.decisions_available():
        impact = fixtures.placeholder_financial()
        for a in impact.assumptions:
            if a.key == "recovery_rate":
                a.value = req.recovery_rate
            elif a.key == "cost_per_false_review":
                a.value = req.cost_per_false_review
            elif a.key == "cost_per_false_block_ratio":
                a.value = req.cost_per_false_block_ratio
        return impact

    return FinancialImpact.model_validate(
        financial_engine.recompute(
            financial_engine.Assumptions(
                recovery_rate=req.recovery_rate,
                cost_per_false_review=req.cost_per_false_review,
                cost_per_false_block_ratio=req.cost_per_false_block_ratio,
            )
        )
    )


def _load_simulation() -> dict | None:
    if not SIMULATION_PATH.is_file():
        return None
    return json.loads(SIMULATION_PATH.read_text())


@app.post("/api/simulate/start", response_model=SimulationStartResponse)
def simulate_start() -> SimulationStartResponse:
    script = _load_simulation()
    if script is None:
        return SimulationStartResponse(run_id="SIM-PLACEHOLDER", total_phases=9, seed=42)
    return SimulationStartResponse(
        run_id=script["run_id"], total_phases=9, seed=script["seed"]
    )


@app.get("/api/simulate/step", response_model=SimulationStep)
def simulate_step(phase: int = Query(default=1, ge=1, le=9)) -> SimulationStep:
    """Replay a precomputed step.

    The script is built offline from real artifacts, so the demo runs the same
    way every time. A demo that can fail live is a demo that will.
    """
    script = _load_simulation()
    if script is None:
        return SimulationStep(
            phase=phase,
            phase_name=f"placeholder_phase_{phase}",
            narrative="No simulation built yet. Run `make simulate`.",
            feed_items=[],
            ring=None,
            audit_entries=[],
            financial_delta=None,
            is_final=phase == 9,
        )
    step = next((s for s in script["steps"] if s["phase"] == phase), None)
    if step is None:
        raise HTTPException(status_code=404, detail=f"No step {phase}.")
    try:
        return SimulationStep.model_validate(step)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "artifacts/simulation.json violates the frozen contract. "
                f"Regenerate with `make simulate`. {exc.error_count()} error(s)."
            ),
        ) from exc
