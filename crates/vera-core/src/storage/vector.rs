//! sqlite-vec based vector store for embedding storage and similarity search.
//!
//! Uses the sqlite-vec extension for brute-force KNN vector search.
//! Vectors are stored alongside the metadata DB in the same SQLite file.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, ffi::sqlite3_auto_extension, params};
use sqlite_vec::sqlite3_vec_init;
use zerocopy::IntoBytes;

const PREFIX_RANGE_SQL: &str =
    "SELECT rowid FROM chunk_id_map WHERE chunk_id >= ?1 AND chunk_id < ?2";
const PREFIX_LOWER_BOUND_SQL: &str = "SELECT rowid FROM chunk_id_map WHERE chunk_id >= ?1";

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
                -- `chunk_id` is UNIQUE, which already builds
                -- `sqlite_autoindex_chunk_id_map_1` over the same column. The
                -- explicit index duplicated it exactly and only added a second
                -- B-tree to maintain on every vector insert. Dropped here so
                -- existing databases shed it on next open; the autoindex
                -- serves both the equality lookup and the prefix range scan.
                DROP INDEX IF EXISTS idx_chunk_id_map;",
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

        // `prepare`, not `prepare_cached`: `limit` is interpolated into the
        // text, so every distinct limit is a distinct cache key. Caching these
        // never hits and evicts the statements that do have stable text, since
        // rusqlite's cache is a 16-entry LRU shared by the whole connection.
        let mut stmt = self
            .conn
            .prepare(&format!(
                "SELECT v.rowid, v.distance
                 FROM vec_chunks v
                 WHERE v.embedding MATCH ?1
                 ORDER BY v.distance
                 LIMIT {limit}"
            ))
            .context("failed to prepare vector search")?;

        let hits: Vec<(i64, f64)> = stmt
            .query_map([query.as_bytes()], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
            })
            .context("failed to execute vector search")?
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read vector result")?;

        // Resolve the rowids in one query instead of one per hit. The KNN
        // query above is deliberately left alone: joining `chunk_id_map` into
        // it would risk losing the vec0 KNN optimization.
        let rowids: Vec<i64> = hits.iter().map(|(rowid, _)| *rowid).collect();
        let chunk_ids = self.chunk_ids_for_rowids(&rowids)?;

        hits.iter()
            .map(|(rowid, distance)| {
                let chunk_id = chunk_ids
                    .get(rowid)
                    .with_context(|| format!("failed to map rowid {rowid} to chunk_id"))?
                    .clone();
                Ok(VectorSearchResult {
                    chunk_id,
                    distance: *distance,
                })
            })
            .collect()
    }

    /// Resolve many rowids to their chunk ids in a single query.
    fn chunk_ids_for_rowids(&self, rowids: &[i64]) -> Result<HashMap<i64, String>> {
        if rowids.is_empty() {
            return Ok(HashMap::new());
        }

        let placeholders = std::iter::repeat_n("?", rowids.len())
            .collect::<Vec<_>>()
            .join(",");
        // Text varies with the number of ids, so plain `prepare` here too.
        let mut stmt = self
            .conn
            .prepare(&format!(
                "SELECT rowid, chunk_id FROM chunk_id_map WHERE rowid IN ({placeholders})"
            ))
            .context("failed to prepare chunk id lookup")?;

        let mapped = stmt
            .query_map(rusqlite::params_from_iter(rowids.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("failed to query chunk ids")?
            .collect::<std::result::Result<HashMap<_, _>, _>>()
            .context("failed to collect chunk ids")?;
        Ok(mapped)
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
        self.delete_by_file_prefix_after_scan(prefix, || {})
    }

    fn delete_by_file_prefix_after_scan<F>(&self, prefix: &str, after_scan: F) -> Result<u64>
    where
        F: FnOnce(),
    {
        // A half-open range rather than `LIKE ?1 ESCAPE '\'`. The ESCAPE
        // clause disqualifies SQLite's LIKE-prefix optimization, so the LIKE
        // form scans `chunk_id_map` in full where a range seeks the index. It
        // also removes the need to escape `%` and `_`, since a range has no
        // wildcards to confuse.
        // One transaction for the whole file instead of two commits per row,
        // opened *before* the scan so a concurrent writer on the same database
        // cannot insert a matching row between finding the rowids and deleting
        // them — that row would survive while this reported success.
        //
        // IMMEDIATE rather than the default DEFERRED because this transaction
        // reads and then writes: a deferred transaction starts as a reader and
        // can fail the upgrade with SQLITE_BUSY_SNAPSHOT, which `busy_timeout`
        // does not retry.
        let tx = rusqlite::Transaction::new_unchecked(
            &self.conn,
            rusqlite::TransactionBehavior::Immediate,
        )
        .context("failed to begin prefix delete transaction")?;

        let rows = rowids_with_prefix(&tx, prefix)?;
        after_scan();
        let count = rows.len() as u64;
        if count == 0 {
            return Ok(count);
        }

        {
            let mut delete_vector = tx
                .prepare_cached("DELETE FROM vec_chunks WHERE rowid = ?1")
                .context("failed to prepare vector delete")?;
            let mut delete_mapping = tx
                .prepare_cached("DELETE FROM chunk_id_map WHERE rowid = ?1")
                .context("failed to prepare chunk id delete")?;
            for rowid in &rows {
                delete_vector
                    .execute(params![rowid])
                    .context("failed to delete vector by prefix")?;
                delete_mapping
                    .execute(params![rowid])
                    .context("failed to delete chunk id by prefix")?;
            }
        }
        tx.commit().context("failed to commit prefix delete")?;

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

/// Rowids of every `chunk_id` starting with `prefix`, via a range scan.
///
/// Takes the connection rather than `&self` so the caller can pass an open
/// transaction, keeping the scan atomic with the deletes that follow it.
fn rowids_with_prefix(conn: &Connection, prefix: &str) -> Result<Vec<i64>> {
    // The two predicates stay distinct because `prefix_upper_bound` returns
    // None when nothing can sort above the prefix, and `>= prefix` is then
    // already exact. Only the execution and collection are shared; merging the
    // SQL would cost the index range plan, which is the point of all this.
    let upper = prefix_upper_bound(prefix);
    let (sql, args): (&str, Vec<&dyn rusqlite::ToSql>) = match &upper {
        Some(upper) => (PREFIX_RANGE_SQL, vec![&prefix, upper]),
        None => (PREFIX_LOWER_BOUND_SQL, vec![&prefix]),
    };

    let mut stmt = conn
        .prepare_cached(sql)
        .context("failed to prepare prefix delete query")?;
    let rows = stmt
        .query_map(args.as_slice(), |row| row.get::<_, i64>(0))
        .context("failed to query chunks by prefix")?
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to collect prefix results")?;
    Ok(rows)
}

/// Smallest string that sorts strictly above every string starting with
/// `prefix`, giving the exclusive upper bound of a prefix range scan.
///
/// SQLite's default `BINARY` collation compares TEXT bytewise, and UTF-8 byte
/// order matches code point order, so incrementing the final character is
/// sufficient. Two cases need care: the successor of a code point may land in
/// the surrogate range, which is not a valid `char`, so it is skipped; and a
/// trailing `char::MAX` has no successor at all, so it is dropped and the
/// character before it is incremented instead.
///
/// Returns `None` when no upper bound exists (an empty prefix, or one made
/// entirely of `char::MAX`), in which case `chunk_id >= prefix` is already
/// exact and needs no upper bound.
fn prefix_upper_bound(prefix: &str) -> Option<String> {
    let mut chars: Vec<char> = prefix.chars().collect();
    while let Some(last) = chars.pop() {
        if let Some(next) = next_char(last) {
            let mut bound: String = chars.into_iter().collect();
            bound.push(next);
            return Some(bound);
        }
    }
    None
}

/// Next valid `char` above `c`, skipping the UTF-16 surrogate range.
fn next_char(c: char) -> Option<char> {
    let mut code = c as u32 + 1;
    while code <= char::MAX as u32 {
        if let Some(next) = char::from_u32(code) {
            return Some(next);
        }
        code += 1;
    }
    None
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
    fn search_pairs_each_chunk_id_with_its_own_distance() {
        // The rowid -> chunk_id mapping is now one batched query returning a
        // HashMap, so the results have to be re-projected back into the KNN
        // distance order. Insert so that rowid order (insertion order) is the
        // reverse of distance order, then assert the pairing — not just that
        // distances ascend, which holds for any implementation that emits one
        // result per hit in hit order.
        // Distances must be strictly increasing, or the KNN order between two
        // equidistant vectors is unspecified and the assertion below would be
        // decided by tie-breaking rather than by distance.
        //   near 0.0   mid ~0.894   far ~1.414   (L2 from the query)
        // Inserted in reverse of that order, so rowid order is not distance
        // order and a SQL-ordered result cannot pass by accident.
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("far", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        store.insert("mid", &[0.6, 0.8, 0.0, 0.0]).unwrap();
        store.insert("near", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 3);

        let order: Vec<&str> = results.iter().map(|r| r.chunk_id.as_str()).collect();
        assert_eq!(
            order,
            vec!["near", "mid", "far"],
            "results must come back in distance order, not rowid order"
        );
        assert!(
            results[0].distance < 0.001,
            "the self-match must keep its own near-zero distance, got {}",
            results[0].distance
        );
        for pair in results.windows(2) {
            assert!(
                pair[0].distance < pair[1].distance,
                "distances must strictly ascend, or the order is a tie-break: {:?}",
                results.iter().map(|r| r.distance).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn chunk_id_batch_lookup_handles_4096_bound_parameters() {
        let store = VectorStore::open_in_memory(4).unwrap();
        for index in 1..=4096 {
            store
                .conn
                .execute(
                    "INSERT INTO chunk_id_map (rowid, chunk_id) VALUES (?1, ?2)",
                    params![index, format!("chunk-{index}")],
                )
                .unwrap();
        }

        let rowids: Vec<i64> = (1..=4096).collect();
        let mapped = store.chunk_ids_for_rowids(&rowids).unwrap();

        assert_eq!(mapped.len(), 4096);
        assert_eq!(mapped.get(&1).map(String::as_str), Some("chunk-1"));
        assert_eq!(mapped.get(&4096).map(String::as_str), Some("chunk-4096"));
    }

    #[test]
    fn search_reports_missing_chunk_id_mapping() {
        let store = VectorStore::open_in_memory(4).unwrap();
        store.insert("missing", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        store
            .conn
            .execute(
                "DELETE FROM chunk_id_map WHERE chunk_id = ?1",
                params!["missing"],
            )
            .unwrap();

        let error = store.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap_err();
        assert!(
            error.to_string().contains("failed to map rowid"),
            "{error:#}"
        );
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

    #[test]
    fn delete_by_file_prefix_does_not_treat_wildcards_as_patterns() {
        // `_` matches any character and `%` any sequence in LIKE. A range
        // predicate has no wildcards, but the paths that would have been
        // mis-matched are exactly the ones worth pinning.
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/a_b.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/axb.rs:0", &[0.0, 1.0, 0.0, 0.0]),
            ("src/100%.rs:0", &[0.0, 0.0, 1.0, 0.0]),
            ("src/100pct.rs:0", &[0.0, 0.0, 0.0, 1.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/a_b.rs:").unwrap(), 1);
        assert_eq!(store.delete_by_file_prefix("src/100%.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 2);

        // The literal-looking siblings survive.
        let remaining = store.search(&[0.0, 1.0, 0.0, 0.0], 4).unwrap();
        let ids: Vec<&str> = remaining.iter().map(|r| r.chunk_id.as_str()).collect();
        assert!(ids.contains(&"src/axb.rs:0"), "{ids:?}");
        assert!(ids.contains(&"src/100pct.rs:0"), "{ids:?}");
    }

    #[test]
    fn delete_by_file_prefix_stops_at_the_prefix_boundary() {
        // The upper bound must exclude the next sibling but include every
        // descendant of the prefix, however long.
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/app.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/app.rs:10", &[0.0, 1.0, 0.0, 0.0]),
            ("src/app2.rs:0", &[0.0, 0.0, 1.0, 0.0]),
            ("src/apq.rs:0", &[0.0, 0.0, 0.0, 1.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/app.rs:").unwrap(), 2);
        assert_eq!(store.count().unwrap(), 2);
        let remaining = store.search(&[0.0, 0.0, 1.0, 0.0], 4).unwrap();
        let ids: Vec<&str> = remaining.iter().map(|r| r.chunk_id.as_str()).collect();
        assert!(ids.contains(&"src/app2.rs:0"), "{ids:?}");
        assert!(ids.contains(&"src/apq.rs:0"), "{ids:?}");
    }

    #[test]
    fn delete_by_file_prefix_handles_non_ascii_paths() {
        let store = VectorStore::open_in_memory(4).unwrap();
        let items: Vec<(&str, &[f32])> = vec![
            ("src/café.rs:0", &[1.0, 0.0, 0.0, 0.0]),
            ("src/cafz.rs:0", &[0.0, 1.0, 0.0, 0.0]),
        ];
        store.insert_batch(&items).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/café.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 1);
        let remaining = store.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(remaining[0].chunk_id, "src/cafz.rs:0");
    }

    #[test]
    fn prefix_upper_bound_covers_the_awkward_code_points() {
        assert_eq!(prefix_upper_bound("src/a"), Some("src/b".to_string()));
        assert_eq!(prefix_upper_bound("a:"), Some("a;".to_string()));

        // Successor lands in the surrogate range and must be skipped.
        let below_surrogates = char::from_u32(0xD7FF).unwrap();
        let bound = prefix_upper_bound(&below_surrogates.to_string()).unwrap();
        assert_eq!(bound.chars().next(), char::from_u32(0xE000));

        // A trailing char::MAX has no successor, so the previous character is
        // incremented and the max is dropped.
        let trailing_max = format!("a{}", char::MAX);
        assert_eq!(prefix_upper_bound(&trailing_max), Some("b".to_string()));

        // Nothing sorts above these.
        assert_eq!(prefix_upper_bound(""), None);
        assert_eq!(prefix_upper_bound(&char::MAX.to_string()), None);
    }

    #[test]
    fn prefix_delete_releases_transaction_for_subsequent_writes() {
        // A second connection can write after the prefix delete commits, which
        // confirms that the transaction was committed and released.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open(&path, 4).unwrap();
        store
            .insert_batch(&[("src/a.rs:0", &[1.0, 0.0, 0.0, 0.0][..])])
            .unwrap();

        let other = VectorStore::open(&path, 4).unwrap();

        assert_eq!(store.delete_by_file_prefix("src/a.rs:").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 0);

        // The second connection still works afterwards: the transaction was
        // committed and released, not left open.
        other
            .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
            .unwrap();
        assert_eq!(other.count().unwrap(), 1);
        assert_eq!(other.delete_by_file_prefix("src/a.rs:").unwrap(), 1);
    }

    #[test]
    fn prefix_delete_takes_the_write_lock_before_it_scans() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.db");
        let store = VectorStore::open(&path, 4).unwrap();
        store
            .insert_batch(&[("src/a.rs:0", &[1.0, 0.0, 0.0, 0.0][..])])
            .unwrap();

        let competitor = VectorStore::open(&path, 4).unwrap();
        competitor
            .conn
            .execute_batch("PRAGMA busy_timeout=0")
            .unwrap();

        let deleted = store
            .delete_by_file_prefix_after_scan("src/a.rs:", || {
                let err = competitor
                    .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
                    .expect_err("the prefix transaction must already hold the write lock");
                assert!(
                    err.chain().any(|cause| matches!(
                        cause.downcast_ref::<rusqlite::Error>(),
                        Some(rusqlite::Error::SqliteFailure(failure, _))
                            if matches!(
                                failure.code,
                                rusqlite::ErrorCode::DatabaseBusy
                                    | rusqlite::ErrorCode::DatabaseLocked
                            )
                    )),
                    "competing write should fail because the database is locked: {err:#}"
                );
            })
            .unwrap();

        assert_eq!(deleted, 1);
        assert_eq!(store.count().unwrap(), 0);
        competitor
            .insert_batch(&[("src/a.rs:1", &[0.0, 1.0, 0.0, 0.0][..])])
            .unwrap();
        assert_eq!(competitor.count().unwrap(), 1);
    }

    #[test]
    fn prefix_range_seeks_the_index_instead_of_scanning() {
        // The whole point of the range form: `LIKE ... ESCAPE` cannot use the
        // index, so a plan check is what actually guards this.
        let store = VectorStore::open_in_memory(4).unwrap();
        let mut stmt = store
            .conn
            .prepare(&format!(
                "EXPLAIN QUERY PLAN\n                 {PREFIX_RANGE_SQL}"
            ))
            .unwrap();
        let plan: Vec<String> = stmt
            .query_map(params!["a", "b"], |row| row.get::<_, String>(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let plan = plan.join(" | ");
        assert!(
            plan.contains("SEARCH")
                && (plan.contains("USING COVERING INDEX") || plan.contains("USING INDEX")),
            "prefix range must search using an index: {plan}"
        );
    }
}
