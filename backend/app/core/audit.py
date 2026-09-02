"""Append-only audit log.

Every consequential decision is written here: what the model saw, what it
decided, which policy applied, and what happened as a result.

Two properties make this an audit trail rather than a log file.

APPEND-ONLY IN STRUCTURE, NOT BY CONVENTION. There is no update or delete path
in this module. The only write operation is `append`.

TAMPER-EVIDENT. Each entry stores the hash of the entry before it and a hash of
its own contents. Editing any historical row breaks every hash after it, and
`verify_chain` reports exactly where. A merchant disputing an automated
intervention, or a regulator asking whether a record was altered, gets an answer
rather than an assurance.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URL = f"sqlite:///{ROOT / 'razorshield.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_URL)

GENESIS_HASH = "0" * 64

ACTORS = {"system", "model", "policy_engine", "llm", "merchant"}


class Base(DeclarativeBase):
    pass


class AuditRow(Base):
    __tablename__ = "audit_log"

    seq = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(32), unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    event = Column(String(64), nullable=False)
    actor = Column(String(16), nullable=False)
    subject_id = Column(String(64), nullable=False, index=True)
    input_summary = Column(Text, nullable=False)
    decision = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    policy = Column(String(32), nullable=False)
    result = Column(Text, nullable=False)
    previous_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False)


_engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=_engine, future=True, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def _digest(payload: dict, previous_hash: str) -> str:
    """Hash of the entry's content plus its predecessor.

    sort_keys makes the serialisation canonical, so the same entry always
    produces the same digest regardless of dict ordering.
    """
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{previous_hash}{body}".encode()).hexdigest()


def _iso(ts: datetime) -> str:
    """Canonical timestamp form for hashing.

    SQLite does not preserve timezone, so a value written as UTC-aware reads
    back naive and would rehash differently, breaking the chain on the very
    first verification. Naive values are treated as UTC on both paths.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _content(row_values: dict) -> dict:
    return {
        "entry_id": row_values["entry_id"],
        "timestamp": _iso(row_values["timestamp"]),
        "event": row_values["event"],
        "actor": row_values["actor"],
        "subject_id": row_values["subject_id"],
        "input_summary": row_values["input_summary"],
        "decision": row_values["decision"],
        "reason": row_values["reason"],
        "policy": row_values["policy"],
        "result": row_values["result"],
    }


def append(
    *,
    event: str,
    actor: str,
    subject_id: str,
    input_summary: str,
    decision: str,
    reason: str,
    policy: str = "v1",
    result: str = "ok",
    timestamp: datetime | None = None,
    session: Session | None = None,
) -> dict:
    """The only write path in this module."""
    if actor not in ACTORS:
        raise ValueError(f"unknown actor '{actor}'; expected one of {sorted(ACTORS)}")

    owns_session = session is None
    session = session or SessionLocal()
    try:
        last = session.execute(
            select(AuditRow).order_by(AuditRow.seq.desc()).limit(1)
        ).scalar_one_or_none()
        previous_hash = last.entry_hash if last else GENESIS_HASH
        next_seq = (last.seq if last else 0) + 1

        values = {
            "entry_id": f"AUD-{next_seq:08d}",
            "timestamp": timestamp or datetime.now(timezone.utc),
            "event": event,
            "actor": actor,
            "subject_id": subject_id,
            "input_summary": input_summary,
            "decision": decision,
            "reason": reason,
            "policy": policy,
            "result": result,
        }
        entry_hash = _digest(_content(values), previous_hash)

        row = AuditRow(**values, previous_hash=previous_hash, entry_hash=entry_hash)
        session.add(row)
        session.commit()
        return to_dict(row)
    finally:
        if owns_session:
            session.close()


def to_dict(row: AuditRow) -> dict:
    return {
        "entry_id": row.entry_id,
        "timestamp": _iso(row.timestamp),
        "event": row.event,
        "actor": row.actor,
        "subject_id": row.subject_id,
        "input_summary": row.input_summary,
        "decision": row.decision,
        "reason": row.reason,
        "policy": row.policy,
        "result": row.result,
    }


def entries(subject_id: str | None = None, limit: int = 200) -> list[dict]:
    with SessionLocal() as session:
        stmt = select(AuditRow).order_by(AuditRow.seq.desc()).limit(limit)
        if subject_id:
            stmt = (
                select(AuditRow)
                .where(AuditRow.subject_id == subject_id)
                .order_by(AuditRow.seq.desc())
                .limit(limit)
            )
        return [to_dict(r) for r in session.execute(stmt).scalars()]


def count() -> int:
    with SessionLocal() as session:
        return len(session.execute(select(AuditRow.seq)).all())


def verify_chain() -> dict:
    """Recompute every hash and report the first entry that does not match."""
    with SessionLocal() as session:
        rows = list(
            session.execute(select(AuditRow).order_by(AuditRow.seq)).scalars()
        )

    previous_hash = GENESIS_HASH
    for row in rows:
        if row.previous_hash != previous_hash:
            return {
                "valid": False,
                "checked": len(rows),
                "broken_at": row.entry_id,
                "detail": "chain link does not match the preceding entry",
            }
        expected = _digest(_content(to_dict(row) | {"timestamp": row.timestamp}), previous_hash)
        if expected != row.entry_hash:
            return {
                "valid": False,
                "checked": len(rows),
                "broken_at": row.entry_id,
                "detail": "entry contents do not match its recorded hash",
            }
        previous_hash = row.entry_hash

    return {"valid": True, "checked": len(rows), "broken_at": None, "detail": ""}


def latest_decision(subject_id: str) -> str | None:
    """Current case status, derived from the log rather than stored separately.

    Keeping a mutable status column would create a second source of truth that
    could disagree with the trail. The log is the truth.
    """
    with SessionLocal() as session:
        row = session.execute(
            select(AuditRow)
            .where(AuditRow.subject_id == subject_id)
            .where(AuditRow.actor == "merchant")
            .order_by(AuditRow.seq.desc())
            .limit(1)
        ).scalar_one_or_none()
    return row.decision if row else None
