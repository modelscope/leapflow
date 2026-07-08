"""Persistence layer — trajectory, skill library, session, conversation, and document stores.

Key infrastructure:
- ``ConnectionHolder`` — shared DuckDB connection protocol
- ``LocalConnectionHolder`` — in-process holder for single-instance mode
- ``connect()`` — lock-aware DuckDB connection factory
- ``DatabaseLockedError`` — clear error for multi-instance lock conflicts
"""

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.conversation_store import DuckDBConversationStore
from leapflow.storage.duckdb_connect import DatabaseLockedError, connect, is_lock_error
from leapflow.storage.session_store import LearningSessionStore
from leapflow.storage.skill_docs import SkillDocStore
from leapflow.storage.skill_library import SkillLibraryStore
from leapflow.storage.trajectory_store import TrajectoryStore
from leapflow.storage.write_buffer import WriteBuffer, execute_with_retry

__all__ = [
    "ConnectionHolder",
    "DatabaseLockedError",
    "DuckDBConversationStore",
    "LocalConnectionHolder",
    "LearningSessionStore",
    "SkillDocStore",
    "SkillLibraryStore",
    "TrajectoryStore",
    "WriteBuffer",
    "connect",
    "execute_with_retry",
    "is_lock_error",
]
