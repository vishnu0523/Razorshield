"""Phase 10 gates: is the audit trail trustworthy?

test_editing_history_breaks_the_chain is what makes this an audit trail rather
than a log file. Without it, "every decision is logged" is a claim nobody can
check.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest
from sqlalchemy import update


@pytest.fixture()
def log(tmp_path, monkeypatch):
    """A fresh database per test, so ordering between tests cannot matter."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'audit.db'}")
    from backend.app.core import audit as module

    audit = importlib.reload(module)
    audit.init_db()
    return audit


def write(audit, n: int = 3, subject: str = "AR-0001") -> None:
    for i in range(n):
        audit.append(
            event=f"event_{i}",
            actor="model",
            subject_id=subject,
            input_summary="feature vector",
            decision=f"risk {50 + i}",
            reason="shared devices across accounts",
        )


# --------------------------------------------------------------------------- #
# Tamper evidence
# --------------------------------------------------------------------------- #


def test_a_fresh_chain_verifies(log) -> None:
    write(log, 5)
    result = log.verify_chain()
    assert result["valid"] is True
    assert result["checked"] == 5


def test_editing_history_breaks_the_chain(log) -> None:
    """Altering any historical row invalidates it and everything after."""
    write(log, 5)
    with log.SessionLocal() as session:
        session.execute(
            update(log.AuditRow)
            .where(log.AuditRow.seq == 2)
            .values(decision="risk 0")
        )
        session.commit()

    result = log.verify_chain()
    assert result["valid"] is False
    assert result["broken_at"] == "AUD-00000002", (
        "the chain must name the exact entry that was altered"
    )


def test_deleting_history_breaks_the_chain(log) -> None:
    write(log, 5)
    with log.SessionLocal() as session:
        session.execute(
            update(log.AuditRow).where(log.AuditRow.seq == 3).values(previous_hash="0" * 64)
        )
        session.commit()
    assert log.verify_chain()["valid"] is False


def test_timestamps_survive_a_round_trip(log) -> None:
    """SQLite drops timezone information.

    A UTC-aware value written and read back naive rehashes differently and would
    break the chain on the first verification. Regression test.
    """
    log.append(
        event="e",
        actor="system",
        subject_id="AR-1",
        input_summary="a",
        decision="b",
        reason="c",
        timestamp=datetime.now(timezone.utc),
    )
    assert log.verify_chain()["valid"] is True


# --------------------------------------------------------------------------- #
# Append-only
# --------------------------------------------------------------------------- #


def test_module_exposes_no_mutation_path(log) -> None:
    """Append-only by structure, not by convention."""
    exported = {n for n in dir(log) if not n.startswith("_")}
    for forbidden in ("update", "delete", "edit", "amend", "remove"):
        assert forbidden not in exported


def test_entry_ids_are_sequential_and_unique(log) -> None:
    write(log, 6)
    ids = [e["entry_id"] for e in log.entries()]
    assert len(set(ids)) == 6
    assert sorted(ids, reverse=True) == ids


def test_unknown_actor_is_rejected(log) -> None:
    """Actor is a closed set: 'who decided this' must be answerable."""
    with pytest.raises(ValueError, match="unknown actor"):
        log.append(
            event="e", actor="somebody", subject_id="X",
            input_summary="a", decision="b", reason="c",
        )


# --------------------------------------------------------------------------- #
# Case status is derived, never stored
# --------------------------------------------------------------------------- #


def test_case_status_comes_from_the_log(log) -> None:
    write(log, 2)
    assert log.latest_decision("AR-0001") is None

    log.append(
        event="merchant_decision", actor="merchant", subject_id="AR-0001",
        input_summary="recommended MANUAL_REVIEW", decision="APPROVED",
        reason="confirmed",
    )
    assert log.latest_decision("AR-0001") == "APPROVED"


def test_a_later_decision_supersedes_without_erasing(log) -> None:
    """Changing your mind appends; it does not rewrite what you decided before."""
    for decision in ("APPROVED", "DISMISSED"):
        log.append(
            event="merchant_decision", actor="merchant", subject_id="AR-0001",
            input_summary="x", decision=decision, reason="y",
        )
    assert log.latest_decision("AR-0001") == "DISMISSED"
    decisions = [e["decision"] for e in log.entries("AR-0001")]
    assert "APPROVED" in decisions, "the earlier decision was erased"
    assert log.verify_chain()["valid"] is True


def test_entries_can_be_filtered_by_subject(log) -> None:
    write(log, 3, subject="AR-0001")
    write(log, 2, subject="AR-0002")
    assert len(log.entries("AR-0001")) == 3
    assert len(log.entries("AR-0002")) == 2
    assert len(log.entries()) == 5


def test_every_entry_matches_the_contract_shape(log) -> None:
    write(log, 2)
    required = {
        "entry_id", "timestamp", "event", "actor", "subject_id",
        "input_summary", "decision", "reason", "policy", "result",
    }
    for entry in log.entries():
        assert set(entry) == required
