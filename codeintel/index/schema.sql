-- SQLite schema for CodeIntel persistent symbol and reference index
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  lang TEXT NOT NULL,
  mtime_ns BIGINT NOT NULL,
  size_bytes BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
  path TEXT NOT NULL,
  lang TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  qualname TEXT NOT NULL,
  start_line INT NOT NULL,
  end_line INT NOT NULL,
  signature TEXT,
  docstring TEXT,
  PRIMARY KEY (path, start_line, kind, name)
);
CREATE TABLE IF NOT EXISTS refs (
  path TEXT NOT NULL,
  lang TEXT NOT NULL,
  kind TEXT NOT NULL,
  src_qualname TEXT NOT NULL,
  dst_qualname TEXT,
  line INT NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_src ON refs(src_qualname);
CREATE INDEX IF NOT EXISTS refs_dst ON refs(dst_qualname);

