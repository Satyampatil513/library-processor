"""SQLite store: frames.db (step 3), scene graph, model calls, verdict log, HITL queue, exemplars."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, path TEXT, country TEXT, status TEXT, created REAL DEFAULT (strftime('%s','now')));
-- step 3: frames.db, each frame aligned to nearest stills and transcript words
CREATE TABLE IF NOT EXISTS frames(
  session_id TEXT, idx INTEGER, ar_timestamp REAL, rgb TEXT, tracking TEXT, blur REAL,
  nearest_still INTEGER, PRIMARY KEY(session_id, idx));
CREATE TABLE IF NOT EXISTS words(
  session_id TEXT, start REAL, end REAL, word TEXT);            -- times in AR-timestamp seconds
CREATE TABLE IF NOT EXISTS objects(
  id TEXT PRIMARY KEY, session_id TEXT, region_id TEXT, box TEXT, pieces TEXT, overlaps TEXT,
  barcode TEXT, crops TEXT, is_book INTEGER, title TEXT, author TEXT, is_old INTEGER, description TEXT,
  isbn TEXT, list_price REAL, currency TEXT, price_concept TEXT, price_source TEXT,
  price_note TEXT, status TEXT, provenance TEXT);
CREATE TABLE IF NOT EXISTS model_calls(
  id INTEGER PRIMARY KEY AUTOINCREMENT, region_id TEXT, model TEXT, output TEXT, tag TEXT);
CREATE TABLE IF NOT EXISTS verdicts(                             -- [F13] logged Jev calls
  id INTEGER PRIMARY KEY AUTOINCREMENT, region_id TEXT, object_id TEXT, field TEXT,
  winner TEXT, confidence REAL, reasoning TEXT, stage TEXT, ts REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS hitl(                                 -- [F16]
  id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT, reason TEXT, payload TEXT,
  status TEXT DEFAULT 'open', resolution TEXT);
CREATE TABLE IF NOT EXISTS exemplars(                            -- [F13/F14]
  id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, payload TEXT, ts REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS gold(
  id INTEGER PRIMARY KEY AUTOINCREMENT, object_id TEXT, truth TEXT);
CREATE TABLE IF NOT EXISTS gold_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, agreement REAL, paused INTEGER, ts REAL DEFAULT (strftime('%s','now')));
CREATE TABLE IF NOT EXISTS learning_state(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS floorplans(session_id TEXT PRIMARY KEY, data TEXT);
"""


class DB:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def x(self, sql: str, args=()):
        cur = self.conn.execute(sql, args)
        self.conn.commit()
        return cur

    def q(self, sql: str, args=()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def upsert(self, table: str, row: dict):
        cols = ",".join(row)
        ph = ",".join("?" * len(row))
        self.x(f"INSERT OR REPLACE INTO {table}({cols}) VALUES({ph})",
               [json.dumps(v) if isinstance(v, (dict, list)) else v for v in row.values()])
