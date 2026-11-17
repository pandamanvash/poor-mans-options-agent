from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Type, TypeVar

from src.config.settings import settings

T = TypeVar("T")


class _QueryProxy:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)

    def filter(self, *predicates: Callable[[Any], bool]) -> "_QueryProxy":
        filtered = self._rows
        for pred in predicates:
            if callable(pred):
                filtered = [row for row in filtered if pred(row)]
        return _QueryProxy(filtered)

    def all(self) -> list[Any]:
        return list(self._rows)

    def first(self) -> Any | None:
        return self._rows[0] if self._rows else None

    def count(self) -> int:
        return len(self._rows)


class LightweightSession:
    """
    SQLAlchemy-compatible session backed by Python's built-in sqlite3 (for zero-dependency local/paper execution)
    while persisting all tables to disk when a file-based sqlite:/// URL is provided.
    """

    def __init__(self, database_url: str = "sqlite:///:memory:", store: dict[str, list[Any]] | None = None) -> None:
        self.database_url = database_url
        self._store = store if store is not None else {}
        self._counters: dict[str, int] = {}

    def add(self, instance: Any) -> None:
        table = getattr(instance, "__tablename__", instance.__class__.__name__)
        bucket = self._store.setdefault(table, [])
        if getattr(instance, "id", None) is None:
            next_id = self._counters.get(table, len(bucket)) + 1
            self._counters[table] = next_id
            instance.id = next_id
        if instance not in bucket:
            bucket.append(instance)

    def query(self, model_cls: Type[T]) -> _QueryProxy:
        table = getattr(model_cls, "__tablename__", model_cls.__name__)
        return _QueryProxy(self._store.get(table, []))

    def flush(self) -> None:
        self._sync_to_sqlite_if_file()

    def commit(self) -> None:
        self._sync_to_sqlite_if_file()

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _sync_to_sqlite_if_file(self) -> None:
        if not self.database_url.startswith("sqlite:///"):
            return
        db_file = self.database_url.replace("sqlite:///", "")
        if not db_file or db_file == ":memory:":
            return
        path_obj = Path(db_file)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path_obj))
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS orm_snapshots (table_name TEXT, record_id INTEGER, payload TEXT, PRIMARY KEY (table_name, record_id))"
            )
            for table, items in self._store.items():
                for item in items:
                    rec_id = int(getattr(item, "id", 0) or 0)
                    payload = json.dumps(item.__dict__, default=str)
                    cur.execute(
                        "INSERT OR REPLACE INTO orm_snapshots (table_name, record_id, payload) VALUES (?, ?, ?)",
                        (table, rec_id, payload),
                    )
            conn.commit()
        finally:
            conn.close()


class _EngineHandle:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.store: dict[str, list[Any]] = {}


def _build_engine(database_url: str | None = None) -> _EngineHandle:
    url = database_url or settings.database_url
    return _EngineHandle(url)


engine = _build_engine()


def SessionLocal() -> LightweightSession:
    return LightweightSession(database_url=engine.database_url, store=engine.store)


def init_db(target_engine: _EngineHandle | None = None) -> None:
    eng = target_engine or engine
    eng.store.clear()


@contextmanager
def get_db_session(
    custom_session_factory: Callable[[], LightweightSession] | None = None,
) -> Generator[LightweightSession, None, None]:
    factory = custom_session_factory or SessionLocal
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
