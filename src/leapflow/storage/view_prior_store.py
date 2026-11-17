# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded deduplicated observation trials on the shared profile DuckDB owner."""
from __future__ import annotations

import json
from typing import Any, Mapping

from leapflow.hardware.view_priors import ViewPriorSettings
from leapflow.storage.connection import ConnectionHolder


class ViewPriorStore:
    def __init__(self, holder: ConnectionHolder, settings: ViewPriorSettings) -> None:
        self._holder = holder
        self.settings = settings
        self._ready = False

    def _initialize(self) -> None:
        if not self._ready:
            self._holder.connection.execute('''CREATE TABLE IF NOT EXISTS observation_view_trials (
                event_id VARCHAR PRIMARY KEY, workcell VARCHAR NOT NULL, scope VARCHAR NOT NULL,
                observed_at DOUBLE NOT NULL, trials VARCHAR NOT NULL, terminal VARCHAR DEFAULT '{}')''')
            self._holder.connection.execute(
                "ALTER TABLE observation_view_trials ADD COLUMN IF NOT EXISTS terminal VARCHAR DEFAULT '{}'")
            self._ready = True

    def accept(self, value: Mapping[str, Any], now: float) -> bool:
        """Only the filtered subscriber calls this; duplicate IDs never multiply support."""
        self._initialize()
        connection = self._holder.connection
        connection.execute('BEGIN TRANSACTION')
        try:
            exists = connection.execute('SELECT 1 FROM observation_view_trials WHERE event_id=?',
                                        [value['event_id']]).fetchone()
            if exists:
                connection.execute('COMMIT')
                return False
            terminal = {k: value[k] for k in ('status', 'failure_code', 'moves') if k in value}
            connection.execute('''INSERT INTO observation_view_trials
                (event_id, workcell, scope, observed_at, trials, terminal) VALUES (?, ?, ?, ?, ?, ?)''',
                [value['event_id'], value['workcell'], value['scope'], value['observed_at'],
                 json.dumps(value['trials'], allow_nan=False), json.dumps(terminal, allow_nan=False)])
            connection.execute('DELETE FROM observation_view_trials WHERE observed_at < ?',
                               [now-self.settings.ttl_s])
            connection.execute('''DELETE FROM observation_view_trials WHERE event_id IN (
                SELECT event_id FROM observation_view_trials ORDER BY observed_at DESC, event_id
                OFFSET ?)''', [self.settings.max_events])
            connection.execute('COMMIT')
            return True
        except Exception:
            connection.execute('ROLLBACK')
            raise

    def hints(self, workcell: str, now: float) -> dict[str, Any]:
        """Restart-safe priors. The node must compare this scope with its live scope."""
        self._initialize()
        rows = self._holder.connection.execute('''SELECT scope, trials FROM observation_view_trials
            WHERE workcell=? AND observed_at >= ? ORDER BY observed_at DESC, event_id''',
            [workcell, now-self.settings.ttl_s]).fetchall()
        if not rows:
            return {}
        scope = rows[0][0]
        totals: dict[str, list[float]] = {}
        for configuration, encoded in rows:
            if configuration != scope:
                continue
            for trial in json.loads(encoded):
                if trial['reward'] is None:
                    continue
                totals.setdefault(trial['feature'], []).append(trial['reward'])
        scores = {feature: sum(values)/len(values) for feature, values in totals.items()}
        ranked = sorted(scores, key=lambda f: (-scores[f], f))[:64]
        return {'prior_scope': scope, 'priors': {f: scores[f] for f in ranked}}
