"""Shared test setup.

Nothing here should be test logic -- only the plumbing every test file
implicitly depends on but shouldn't have to arrange for itself.

Tables must exist before any test touches the database. Previously nothing
guaranteed that: init_db() only runs inside FastAPI's startup event, and only
one test in the whole suite (test_live_credentials_are_rejected_at_startup)
happens to trigger that event via `with TestClient(app):`. Every other test
uses a bare `TestClient(app)`, which does not run startup. This worked only
because razorshield.db already had tables from an earlier pipeline run or app
launch sitting on disk -- delete that file, or clone the repo fresh and run
pytest before ever starting the app, and whichever test imports first fails
with "no such table", for a reason that has nothing to do with what it's
testing. session-scoped and autouse so it runs once, unconditionally, before
collection order can matter.
"""

from __future__ import annotations

import pytest

from backend.app.core import audit


@pytest.fixture(scope="session", autouse=True)
def _database_ready() -> None:
    audit.init_db()
