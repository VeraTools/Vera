//! sqlite-vec based vector store for embedding storage and similarity search.
//!
//! Uses the sqlite-vec extension for brute-force KNN vector search.
//! Vectors are stored alongside the metadata DB in the same SQLite file.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, ffi::sqlite3_auto_extension, params};
use sqlite_vec::sqlite3_vec_init;
use zerocopy::IntoBytes;

/// sqlite-vec backed vector store for embedding search.
pub struct VectorStore {
    conn: Connection,
    dim: usize,
}

/// Maximum `k` sqlite-vec accepts in a KNN query. Requesting more is a hard
/// error from the extension, not a soft limit.
///
/// Public so callers can size their candidate pools against the real ceiling
/// instead of scaling past it and relying on [`VectorStore::search`] to clamp.
pub const MAX_KNN_K: usize = 4096;

/// A single vector search result: chunk ID and distance score.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// The chunk ID (rowid in the vec table, mapped to chunk string ID).
    pub chunk_id: String,
    /// Distance from the query vector (lower is closer).
    pub distance: f64,
}

impl VectorStore {
    /// Open (or create) a vector store at the given path.
    ///
    /// The `dim` parameter specifies the vector dimensionality.
    pub fn open(db_path: &std::path::Path, dim: usize) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open(db_path)
            .with_context(|| format!("failed to open vector db: {}", db_path.display()))?;
        let store = Self { conn, dim };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory vector store (useful for testing).
    pub fn open_in_memory(dim: usize) -> Result<Self> {
        register_sqlite_vec();
        let conn = Connection::open_in_memory().context("failed to open in-memory vector db")?;
        let store = Self { conn, dim };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the vector table schema.
    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .context("failed to set vector db pragmas")?;

        // Mapping from string chunk IDs to integer rowids.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS chunk_id_map (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_chunk_id_map
                    ON chunk_id_map(chunk_id);",
            )
            .context("failed to create chunk_id_map table")?;

        // sqlite-vec virtual table for vector storage.
        self.conn
            .execute_batch(&format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                 USING vec0(embedding float[{}])",
                self.dim
            ))
            .context("failed to create vec_chunks virtual table")?;

        Ok(())
    }

    /// Insert a single vector for a chunk.
    ///
    /// Uses INSERT OR IGNORE + SELECT to get a stable rowid, avoiding the
    /// AUTOINCREMENT orphan problem where INSERT OR REPLACE allocates a
    /// new rowid and orphans old vectors in the vec_chunks virtual table.
    /// For re-inserts (same chunk_id), deletes the old vector first since
    /// the vec0 virtual table does not support INSERT OR REPLACE.
    pub fn insert(&self, chunk_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            anyhow::bail!(
                "vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            );
        }

        // Use INSERT OR IGNORE to preserve existing rowid if already present.
        self.conn
            .execute(
                "INSERT OR IGNORE INTO chunk_id_map (chunk_id) VALUES (?1)",
                params![chunk_id],
            )
            .context("failed to insert chunk id mapping")?;

        let rowid: i64 = self
            .conn
            .query_row(
                "SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .context("failed to get rowid for chunk")?;

        // Delete any existing vector for this rowid before inserting.
        // vec0 virtual tables do not support INSERT OR REPLACE.
        self.conn
            .execute("DELETE FROM vec_chunks WHERE rowid = ?1", params![rowid])
            .ok(); // Ignore error if row doesn't exist.

        self.conn
            .execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, vector.as_bytes()],
            )
            .context("failed to insert vector")?;

        Ok(())
    }

    /// Insert a batch of vectors.
    ///
    /// Uses INSERT OR IGNORE to preserve stable rowids, avoiding the
    /// AUTOINCREMENT orphan problem. For re-inserts, deletes old vectors
    /// first since the vec0 virtual table doesn't support upsert.
    pub fn insert_batch(&self, items: &[(&str, &[f32])]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin vector insert transaction")?;
        {
            let mut id_stmt = self
                .conn
                .prepare_cached("INSERT OR IGNORE INTO chunk_id_map (chunk_id) VALUES (?1)")
                .context("failed to prepare id insert")?;

            let mut rowid_stmt = self
                .conn
                .prepare_cached("SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1")
                .context("failed to prepare rowid query")?;

            let mut del_vec_stmt = self
                .conn
                .prepare_cached("DELETE FROM vec_chunks WHERE rowid = ?1")
                .context("failed to prepare vector delete")?;

            let mut vec_stmt = self
                .conn
                .prepare_cached("INSERT INTO vec_chunks (rowid, embedding) VALUES (?1, ?2)")
                .context("failed to prepare vector insert")?;

            for (chunk_id, vector) in items {
                if vector.len() != self.dim {
                    anyhow::bail!(
                        "vector dimension mismatch for {}: expected {}, got {}",
                        chunk_id,
                        self.dim,
                        vector.len()
                    );
                }

                id_stmt
                    .execute(params![chunk_id])
                    .context("failed to insert chunk id")?;

                let rowid: i64 = rowid_stmt
                    .query_row(params![chunk_id], |row| row.get(0))
                    .context("failed to get rowid")?;

                // Delete old vector if exists (vec0 doesn't support upsert).
                del_vec_stmt.execute(params![rowid]).ok();

                vec_stmt
                    .execute(params![rowid, vector.as_bytes()])
                    .context("failed to insert vector")?;
            }
        }
        tx.commit().context("failed to commit vector batch")?;
        Ok(())
    }

    /// Find the nearest neighbors to a query vector.
    ///
    /// Returns up to `limit` results sorted by ascending distance.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorSearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!(
                "query vector dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            );
        }

        // sqlite-vec reads this LIMIT as the KNN `k` and rejects anything above
        // MAX_KNN_K with "k value in knn query too large". Callers scale the
        // candidate pool from the query type and result limit, which can exceed
        // it on natural-language queries, and the whole vector arm would then be
        // dropped in favour of BM25-only results. Ask for as many as the backend
        // allows instead.
        // Warn rather than debug: this is a silent quality reduction, and the
        // reason #38 was hard to diagnose was a swallowed signal. Vera's own
        // retrieval path now bounds the pool before it gets here, so reaching
        // this branch means an external caller asked for more than the backend
        // can give.
        if limit > MAX_KNN_K {
            tracing::warn!(
                requested = limit,
                clamped = MAX_KNN_K,
                "clamping vector search limit to the sqlite-vec KNN cap; \
                 the extra candidates are not fetched"
            );
        }
        let limit = limit.min(MAX_KNN_K);

        let mut stmt = self
            .conn
            .prepare_cached(&format!(
                "SELECT v.rowid, v.distance
                 FROM vec_chunks v
                 WHERE v.embedding MATCH ?1
                 ORDER BY v.distance
                 LIMIT {limit}"
            ))
            .context("failed to prepare vector search")?;

        let rows = stmt
            .query_map([query.as_bytes()], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .context("failed to execute vector search")?;

        let mut results = Vec::new();
        for row in rows {
            let (rowid, distance) = row.context("failed to read vector result")?;
            // Map rowid back to chunk_id.
            let chunk_id: String = self
                .conn
                .query_row(
                    "SELECT chunk_id FROM chunk_id_map WHERE rowid = ?1",
                    params![rowid],
                    |row| row.get(0),
                )
                .with_context(|| format!("failed to map rowid {rowid} to chunk_id"))?;

            results.push(VectorSearchResult { chunk_id, distance });
        }

        Ok(results)
    }

    /// Count total vectors in the store.
    pub fn count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunk_id_map", [], |row| row.get(0))
            .context("failed to count vectors")?;
        Ok(count as u64)
    }

    /// Delete a vector by chunk ID.
    pub fn delete(&self, chunk_id: &str) -> Result<bool> {
        let rowid: Option<i64> = self
            .conn
            .query_row(
                "SELECT rowid FROM chunk_id_map WHERE chunk_id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .optional()
            .context("failed to look up chunk for deletion")?;

        if let Some(rowid) = rowid {
            self.conn
                .execute("DELETE FROM vec_chunks WHERE rowid = ?1", params![rowid])
                .context("failed to delete vector")?;
            self.conn
                .execute("DELETE FROM chunk_id_map WHERE rowid = ?1", params![rowid])
                .context("failed to delete chunk id mapping")?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete all vectors whose chunk_id starts with the given prefix.
    ///
    /// This is used for incremental indexing: when a file is re-indexed, all
    /// old chunks for that file (whose IDs start with "filepath:") are removed.
    pub fn delete_by_file_prefix(&self, prefix: &str) -> Result<u64> {
        // Find all rowids matching the prefix.
        // Escape SQL LIKE wildcards (_ and %) in the prefix to avoid
        // incorrect matches on file paths containing those characters.
        let mut stmt = self
            .conn
            .prepare(
                "SELECT rowid, chunk_id FROM chunk_id_map \
                 WHERE chunk_id LIKE ?1 ESCAPE '\\'",
            )
            .context("failed to prepare prefix delete query")?;

        let escaped_prefix = prefix
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let like_pattern = format!("{escaped_prefix}%");
        let rows: Vec<(i64, String)> = stmt
            .query_map(params![like_pattern], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("failed to query chunks by prefix")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to collect prefix results")?;

        let count = rows.len() as u64;
        for (rowid, _chunk_id) in &rows {
            self.conn
                .execute("DELETE FROM vec_chunks WHERE rowid = ?1", params![rowid])
                .context("failed to delete vector by prefix")?;
            self.conn
                .execute("DELETE FROM chunk_id_map WHERE rowid = ?1", params![rowid])
                .context("failed to delete chunk id by prefix")?;
        }

        Ok(count)
    }

    /// Clear all vectors from the store.
    pub fn clear(&self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM vec_chunks; DELETE FROM chunk_id_map;")
            .context("failed to clear vector store")?;
        Ok(())
    }

    /// Get the configured vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Register the sqlite-vec extension globally (idempotent).
fn register_sqlite_vec() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        unsafe {
            // sqlite-vec requires registering via auto_extension with a transmute
            // from the C-style init function pointer to the sqlite3 extension type.
            #[allow(clippy::missing_transmute_annotations)]
            let func = std::mem::transmute(sqlite3_vec_init as *const ());
            sqlite3_auto_extension(Some(func));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random for testing.
        let mut v = Vec::with_capacity(dim);
        let mut s = seed;
        for _ in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.push(((s >> 33) as f32) / (u32::MAX as f32));
        }
        v
    }

    #[test]
    fn insert_and_count() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("chunk1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("chunk2", &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let result = store.insert("chunk1", &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("dimension mismatch")
        );
    }

    #[test]
    fn nearest_neighbor_self_query() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        store.insert("c1", &v1).unwrap();
        store.insert("c2", &v2).unwrap();
        store.insert("c3", &v3).unwrap();

        // Query with v1 should return c1 as the closest.
        let results = store.search(&v1, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001); // Self-match should be ~0 distance.
    }

    #[test]
    fn search_clamps_limit_above_sqlite_vec_knn_cap() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            store.insert(&format!("c{i}"), &v).unwrap();
        }

        // Without clamping, sqlite-vec fails the query outright with
        // "k value in knn query too large".
        let results = store
            .search(&[5.0, 0.0, 0.0, 0.0], MAX_KNN_K + 1)
            .expect("oversized k must be clamped, not rejected");
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn search_respects_limit() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            store.insert(&format!("c{i}"), &v).unwrap();
        }

        let results = store.search(&[5.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn delete_vector() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("c2", &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(store.count().unwrap(), 2);

        assert!(store.delete("c1").unwrap());
        assert_eq!(store.count().unwrap(), 1);

        // Deleting non-existent returns false.
        assert!(!store.delete("nonexistent").unwrap());
    }

    #[test]
    fn clear_vectors() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        store.insert("c2", &[5.0, 6.0, 7.0, 8.0]).unwrap();

        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn batch_insert() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("c1", &[1.0, 0.0, 0.0, 0.0]),
            ("c2", &[0.0, 1.0, 0.0, 0.0]),
            ("c3", &[0.0, 0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 3);
    }

    #[test]
    fn higher_dim_vectors_work() {
        // Validate with 4096-dim (Qwen3 production dimensionality).
        let dim = 4096;
        let store = VectorStore::open_in_memory(dim).unwrap();

        let v1 = random_vector(dim, 42);
        let v2 = random_vector(dim, 123);
        let v3 = random_vector(dim, 456);

        store.insert("c1", &v1).unwrap();
        store.insert("c2", &v2).unwrap();
        store.insert("c3", &v3).unwrap();

        assert_eq!(store.count().unwrap(), 3);

        // Self-query should find the same vector.
        let results = store.search(&v1, 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[test]
    fn query_dimension_mismatch_rejected() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("c1", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = store.search(&[1.0, 2.0], 1);
        assert!(result.is_err());
    }

    #[test]
    fn insert_same_chunk_preserves_rowid_no_orphans() {
        // Verify that re-inserting a chunk_id updates the vector in-place
        // without creating orphaned rows (the AUTOINCREMENT fix).
        let store = VectorStore::open_in_memory(4).unwrap();

        // Insert initial vector.
        store.insert("c1", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(store.count().unwrap(), 1);

        // Re-insert same chunk_id with a different vector.
        store.insert("c1", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            store.count().unwrap(),
            1,
            "count should still be 1 after re-insert"
        );

        // Search should find the updated vector, not the old one.
        let results = store.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001, "should match updated vector");

        // Old vector should not be a close match.
        let results_old = store.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        // The only result should be c1, and it should have nonzero distance
        // from the old vector since we updated it.
        assert_eq!(results_old.len(), 1);
        assert!(
            results_old[0].distance > 0.5,
            "old vector should not match closely"
        );
    }

    #[test]
    fn batch_insert_same_chunk_no_orphans() {
        let store = VectorStore::open_in_memory(4).unwrap();

        // Insert initial batch.
        let items: Vec<(&str, &[f32])> =
            vec![("c1", &[1.0, 0.0, 0.0, 0.0]), ("c2", &[0.0, 1.0, 0.0, 0.0])];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 2);

        // Re-insert c1 with updated vector.
        let items2: Vec<(&str, &[f32])> = vec![("c1", &[0.0, 0.0, 1.0, 0.0])];
        store.insert_batch(&items2).unwrap();
        assert_eq!(store.count().unwrap(), 2, "count should still be 2");

        // Verify c1 has the updated vector.
        let results = store.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "c1");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn delete_by_file_prefix() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/main.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/main.rs:1", &[0.0, 1.0, 0.0, 0.0]),
            ("src/lib.rs:0", &[0.0, 0.0, 1.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();
        assert_eq!(store.count().unwrap(), 3);

        // Delete all vectors for src/main.rs.
        let deleted = store.delete_by_file_prefix("src/main.rs:").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.count().unwrap(), 1);

        // Remaining vector should be src/lib.rs:0.
        let results = store.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].chunk_id, "src/lib.rs:0");
    }
}
