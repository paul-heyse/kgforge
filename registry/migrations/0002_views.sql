CREATE OR REPLACE VIEW dense_vectors_view AS
SELECT vec.*
FROM dense_runs AS runs,
     read_parquet(runs.parquet_root || '/*/*.parquet', union_by_name=true) AS vec;

CREATE OR REPLACE VIEW splade_vectors_view AS
SELECT vec.*
FROM sparse_runs AS runs,
     read_parquet(runs.parquet_root || '/*/*.parquet', union_by_name=true) AS vec
WHERE runs.backend = 'lucene-impact';

CREATE OR REPLACE VIEW chunk_texts AS
SELECT chunks.*
FROM datasets AS ds,
     read_parquet(ds.parquet_root || '/*/*.parquet', union_by_name=true) AS chunks
WHERE ds.kind = 'chunks';

INSERT OR IGNORE INTO schema_version(version) VALUES (2);
