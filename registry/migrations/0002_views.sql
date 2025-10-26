CREATE OR REPLACE VIEW dense_vectors_view AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM dense_runs), union_by_name=true);

CREATE OR REPLACE VIEW splade_vectors_view AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM sparse_runs WHERE backend='lucene-impact'), union_by_name=true);

CREATE OR REPLACE VIEW chunk_texts AS
SELECT * FROM read_parquet(
  (SELECT parquet_root FROM datasets WHERE kind='chunks'), union_by_name=true);

INSERT OR IGNORE INTO schema_version(version) VALUES (2);
