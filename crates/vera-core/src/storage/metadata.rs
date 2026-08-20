//! SQLite-based metadata store for chunk attributes.
//!
//! Stores chunk metadata (file path, line ranges, language, symbol info)
//! in a SQLite database. Uses WAL mode for concurrent read performance.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};

use crate::parsing::type_relations::{RawTypeRelation, TypeRelationKind};
use crate::types::{Chunk, Language, SymbolType};

/// A call site where a symbol is called from.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CallerRef {
    pub file_path: String,
    pub line: u32,
    pub caller: Option<String>,
}

/// A symbol called by another symbol.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CalleeRef {
    pub file_path: String,
    pub line: u32,
    pub callee: String,
}

/// A symbol with no callers (potential dead code).
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeadSymbol {
    pub symbol_name: String,
    pub file_path: String,
    pub line: u32,
    pub symbol_type: Option<String>,
}

/// An explicit type relation pointing at a target symbol.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TypeRelationRef {
    pub file_path: String,
    pub line: u32,
    pub owner: String,
    pub target: String,
    pub kind: TypeRelationKind,
}

/// Persisted file-level indexing state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileIndexStatus {
    Indexed,
    ParseError,
}

impl FileIndexStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Indexed => "indexed",
            Self::ParseError => "parse_error",
        }
    }

    fn from_db(value: &str) -> std::result::Result<Self, std::io::Error> {
        match value {
            "indexed" => Ok(Self::Indexed),
            "parse_error" => Ok(Self::ParseError),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown file index status: {other}"),
            )),
        }
    }
}

/// File-level indexing state captured during indexing/update.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileIndexState {
    pub file_path: String,
    pub language: String,
    pub status: FileIndexStatus,
    pub tree_has_error: bool,
    pub tier0_fallback: bool,
    pub chunk_count: u64,
}

/// Aggregate index health derived from persisted file states.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IndexHealth {
    pub files_indexed: u64,
    pub files_with_tree_sitter_errors: u64,
    pub files_using_tier0_fallback: u64,
    pub files_with_parse_failures: u64,
    pub by_language: Vec<LanguageHealthStat>,
}

/// Per-language index health derived from persisted file states.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LanguageHealthStat {
    pub language: String,
    pub files_indexed: u64,
    pub files_with_tree_sitter_errors: u64,
    pub files_using_tier0_fallback: u64,
    pub files_with_parse_failures: u64,
}

/// Case-insensitive lookup queries, held as constants so the query-plan tests
/// assert against the exact text the production methods run.
///
/// An expression index only applies when the index expression matches the
/// query text. A test that asserted on its own copy of the SQL would keep
/// passing after the original drifted, and the planner would silently fall
/// back to a full scan with nothing failing.
const SQL_CHUNKS_BY_SYMBOL_NAME: &str =
    "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE lower(symbol_name) = lower(?1)
                 ORDER BY file_path, line_start";

const SQL_FIND_CALLERS: &str = "SELECT file_path, line, caller FROM [references]
                 WHERE lower(callee) = lower(?1)
                 ORDER BY file_path, line";

const SQL_FIND_CALLEES: &str = "SELECT file_path, line, callee FROM [references]
                 WHERE lower(caller) = lower(?1)
                 ORDER BY file_path, line";

const SQL_FIND_TYPE_RELATIONS: &str = "SELECT file_path, line, owner, target, kind
                 FROM type_relations
                 WHERE lower(target) = lower(?1)
                 ORDER BY file_path, line, owner";

const SQL_FIND_DEAD_SYMBOLS: &str = "SELECT c.symbol_name, c.file_path, c.line_start, c.symbol_type
                 FROM chunks c
                 WHERE c.symbol_name IS NOT NULL
                   AND c.symbol_type IN ('function', 'method')
                   AND lower(c.symbol_name) NOT IN ('main', 'new', 'default', 'drop', 'clone', 'fmt', 'from', 'into', 'deref', 'init', 'setup', 'teardown')
                   AND NOT EXISTS (
                       SELECT 1 FROM [references] r
                       WHERE lower(r.callee) = lower(c.symbol_name)
                   )
                 ORDER BY c.file_path, c.line_start";

/// SQLite-backed metadata store for chunk attributes.
pub struct MetadataStore {
    conn: Connection,
}

impl MetadataStore {
    /// Open (or create) a metadata store at the given path.
    pub fn open(db_path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(db_path)
            .with_context(|| format!("failed to open metadata db: {}", db_path.display()))?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Create an in-memory metadata store (useful for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("failed to open in-memory metadata db")?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    /// Initialize the database schema and pragmas.
    fn init_schema(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA foreign_keys=ON;",
            )
            .context("failed to set SQLite pragmas")?;

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT NOT NULL,
                    symbol_type TEXT,
                    symbol_name TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path
                    ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_language
                    ON chunks(language);
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name
                    ON chunks(symbol_name);
                -- `get_chunks_by_symbol_name` matches on `lower(symbol_name)`.
                -- An index on the bare column cannot serve a predicate over an
                -- expression, so without this the lookup scans the table.
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name_lower
                    ON chunks(lower(symbol_name));",
            )
            .context("failed to create chunks table")?;

        // File-level content hashing for incremental indexing.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS file_hashes (
                     file_path TEXT PRIMARY KEY,
                     content_hash TEXT NOT NULL
                 );",
            )
            .context("failed to create file_hashes table")?;

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS file_index_state (
                    file_path TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tree_has_error INTEGER NOT NULL,
                    tier0_fallback INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_file_index_state_language
                    ON file_index_state(language);
                CREATE INDEX IF NOT EXISTS idx_file_index_state_status
                    ON file_index_state(status);",
            )
            .context("failed to create file_index_state table")?;

        // Index metadata (model name, dimensions, etc.)
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );",
            )
            .context("failed to create index_metadata table")?;

        // Call-site references for call graph analysis.
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS [references] (
                    file_path TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    callee TEXT NOT NULL,
                    caller TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_refs_callee
                    ON [references](callee);
                CREATE INDEX IF NOT EXISTS idx_refs_caller
                    ON [references](caller);
                CREATE INDEX IF NOT EXISTS idx_refs_file_path
                    ON [references](file_path);
                -- `find_callers`, `find_callees` and the `NOT EXISTS` subquery
                -- in `find_dead_symbols` all match on `lower(...)`, which an
                -- index on the bare column cannot serve.
                CREATE INDEX IF NOT EXISTS idx_refs_callee_lower
                    ON [references](lower(callee));
                CREATE INDEX IF NOT EXISTS idx_refs_caller_lower
                    ON [references](lower(caller));",
            )
            .context("failed to create references table")?;

        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS type_relations (
                    file_path TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    owner TEXT NOT NULL,
                    target TEXT NOT NULL,
                    kind TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_type_relations_target
                    ON type_relations(target);
                CREATE INDEX IF NOT EXISTS idx_type_relations_owner
                    ON type_relations(owner);
                CREATE INDEX IF NOT EXISTS idx_type_relations_file_path
                    ON type_relations(file_path);
                -- `find_type_relations` matches on `lower(target)`, which an
                -- index on the bare column cannot serve.
                CREATE INDEX IF NOT EXISTS idx_type_relations_target_lower
                    ON type_relations(lower(target));",
            )
            .context("failed to create type_relations table")?;

        Ok(())
    }

    /// Insert a batch of chunks into the store.
    pub fn insert_chunks(&self, chunks: &[Chunk]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin transaction")?;
        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "INSERT OR REPLACE INTO chunks
                     (id, file_path, line_start, line_end, content, language, symbol_type, symbol_name)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                )
                .context("failed to prepare insert statement")?;

            for chunk in chunks {
                let sym_type = chunk.symbol_type.map(|st| st.to_string());
                stmt.execute(params![
                    chunk.id,
                    chunk.file_path,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.content,
                    chunk.language.to_string(),
                    sym_type,
                    chunk.symbol_name,
                ])
                .with_context(|| format!("failed to insert chunk: {}", chunk.id))?;
            }
        }
        tx.commit().context("failed to commit chunk inserts")?;
        Ok(())
    }

    /// Get a chunk by its ID.
    pub fn get_chunk(&self, id: &str) -> Result<Option<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks WHERE id = ?1",
            )
            .context("failed to prepare select statement")?;

        let result = stmt
            .query_row(params![id], |row| Ok(row_to_chunk(row)))
            .optional()
            .context("failed to query chunk by id")?;

        match result {
            Some(chunk) => Ok(Some(chunk?)),
            None => Ok(None),
        }
    }

    /// Get multiple chunks by id in a single query.
    ///
    /// Returns a lookup map keyed by chunk id rather than a `Vec`, because
    /// SQLite does not preserve any particular row order for `IN (...)`
    /// queries. Callers that need results in a specific order (e.g.
    /// vector-distance ranking) must re-project from the map themselves.
    pub fn get_chunks_by_ids(&self, ids: &[String]) -> Result<HashMap<String, Chunk>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }

        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, file_path, line_start, line_end, content, language,
                    symbol_type, symbol_name
             FROM chunks WHERE id IN ({placeholders})"
        );
        let mut stmt = self
            .conn
            .prepare(&sql)
            .context("failed to prepare batch chunk query")?;

        let rows = stmt
            .query_map(rusqlite::params_from_iter(ids.iter()), |row| {
                Ok(row_to_chunk(row))
            })
            .context("failed to query chunks by id batch")?;

        let chunks: Vec<Chunk> = collect_rows(rows)?
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        Ok(chunks.into_iter().map(|c| (c.id.clone(), c)).collect())
    }

    /// Get all chunks for a given file path.
    pub fn get_chunks_by_file(&self, file_path: &str) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks WHERE file_path = ?1
                 ORDER BY line_start",
            )
            .context("failed to prepare file chunks query")?;

        let rows = stmt
            .query_map(params![file_path], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by file")?;

        collect_rows(rows)?.into_iter().collect()
    }

    /// Get all chunks whose symbol name matches exactly (case-insensitive).
    pub fn get_chunks_by_symbol_name(&self, symbol_name: &str) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(SQL_CHUNKS_BY_SYMBOL_NAME)
            .context("failed to prepare symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by symbol name")?;

        collect_rows(rows)?.into_iter().collect()
    }

    /// Get all chunks whose symbol name matches exactly (case-sensitive).
    pub fn get_chunks_by_symbol_name_case_sensitive(
        &self,
        symbol_name: &str,
    ) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE symbol_name = ?1
                 ORDER BY file_path, line_start",
            )
            .context("failed to prepare case-sensitive symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name], |row| Ok(row_to_chunk(row)))
            .context("failed to query chunks by symbol name (case-sensitive)")?;

        collect_rows(rows)?.into_iter().collect()
    }

    /// Get chunks whose symbol names contain the given term (case-insensitive).
    pub fn get_chunks_by_symbol_name_substring(
        &self,
        symbol_name: &str,
        limit: usize,
    ) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, file_path, line_start, line_end, content, language,
                        symbol_type, symbol_name
                 FROM chunks
                 WHERE symbol_name IS NOT NULL
                   AND instr(lower(symbol_name), lower(?1)) > 0
                 ORDER BY file_path, line_start
                 LIMIT ?2",
            )
            .context("failed to prepare substring symbol chunks query")?;

        let rows = stmt
            .query_map(params![symbol_name, limit as i64], |row| {
                Ok(row_to_chunk(row))
            })
            .context("failed to query chunks by symbol name substring")?;

        collect_rows(rows)?.into_iter().collect()
    }

    /// Count total chunks in the store.
    pub fn chunk_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .context("failed to count chunks")?;
        Ok(count as u64)
    }

    /// Delete all chunks associated with a file path.
    pub fn delete_chunks_by_file(&self, file_path: &str) -> Result<u64> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM chunks WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete chunks by file")?;
        Ok(deleted as u64)
    }

    /// Clear all data from the store.
    pub fn clear(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "DELETE FROM chunks; DELETE FROM file_hashes; DELETE FROM file_index_state; DELETE FROM index_metadata; DELETE FROM [references]; DELETE FROM type_relations;",
            )
            .context("failed to clear metadata store")?;
        Ok(())
    }

    /// Insert or replace a batch of file states.
    pub fn insert_file_states(&self, states: &[FileIndexState]) -> Result<()> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin file state transaction")?;
        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "INSERT OR REPLACE INTO file_index_state
                     (file_path, language, status, tree_has_error, tier0_fallback, chunk_count)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .context("failed to prepare file state insert")?;

            for state in states {
                stmt.execute(params![
                    state.file_path,
                    state.language,
                    state.status.as_str(),
                    state.tree_has_error,
                    state.tier0_fallback,
                    state.chunk_count as i64,
                ])
                .with_context(|| format!("failed to insert file state: {}", state.file_path))?;
            }
        }
        tx.commit().context("failed to commit file state inserts")?;
        Ok(())
    }

    /// Insert or replace a single file state.
    pub fn upsert_file_state(&self, state: &FileIndexState) -> Result<()> {
        self.insert_file_states(std::slice::from_ref(state))
    }

    /// Delete file state for a path.
    pub fn delete_file_state(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM file_index_state WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete file state")?;
        Ok(())
    }

    /// Store a key-value pair in index_metadata.
    pub fn set_index_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .context("failed to set index metadata")?;
        Ok(())
    }

    /// Retrieve a key's value from index_metadata.
    pub fn get_index_meta(&self, key: &str) -> Result<Option<String>> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM index_metadata WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .context("failed to get index metadata")?;
        Ok(result)
    }

    /// Store a file content hash for incremental indexing.
    pub fn set_file_hash(&self, file_path: &str, hash: &str) -> Result<()> {
        self.set_file_hashes_batch_borrowed(&[(file_path, hash)])
    }

    /// Get the stored content hash for a file.
    pub fn get_file_hash(&self, file_path: &str) -> Result<Option<String>> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?1",
                params![file_path],
                |row| row.get(0),
            )
            .optional()
            .context("failed to get file hash")?;
        Ok(result)
    }

    /// Delete a file hash entry.
    pub fn delete_file_hash(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM file_hashes WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete file hash")?;
        Ok(())
    }

    /// Get all tracked files, including parse failures that produced no chunks.
    pub fn tracked_files(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT file_path FROM file_hashes ORDER BY file_path")
            .context("failed to prepare tracked files query")?;
        let rows = stmt
            .query_map([], |row| row.get(0))
            .context("failed to query tracked files")?;
        collect_rows(rows)
    }

    /// Get all persisted file states.
    pub fn file_states(&self) -> Result<Vec<FileIndexState>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT file_path, language, status, tree_has_error, tier0_fallback, chunk_count
                 FROM file_index_state
                 ORDER BY file_path",
            )
            .context("failed to prepare file states query")?;
        let rows = stmt
            .query_map([], |row| {
                let status: String = row.get(2)?;
                Ok(FileIndexState {
                    file_path: row.get(0)?,
                    language: row.get(1)?,
                    status: FileIndexStatus::from_db(&status).map_err(|err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            2,
                            rusqlite::types::Type::Text,
                            Box::new(err),
                        )
                    })?,
                    tree_has_error: row.get(3)?,
                    tier0_fallback: row.get(4)?,
                    chunk_count: row.get::<_, i64>(5)? as u64,
                })
            })
            .context("failed to query file states")?;

        collect_rows(rows)
    }

    // ── Reference (call graph) operations ──────────────────────────

    /// Insert a batch of call-site references for a single file.
    pub fn insert_references(
        &self,
        file_path: &str,
        refs: &[crate::parsing::references::RawReference],
    ) -> Result<()> {
        self.insert_parse_artifacts_batch_borrowed(&[(file_path, refs)], &[])
    }

    /// Insert a batch of explicit type relations for a single file.
    pub fn insert_type_relations(
        &self,
        file_path: &str,
        relations: &[RawTypeRelation],
    ) -> Result<()> {
        self.insert_parse_artifacts_batch_borrowed(&[], &[(file_path, relations)])
    }

    /// Store content hashes for many files in a single transaction.
    ///
    /// Equivalent to calling [`Self::set_file_hash`] once per file, but
    /// issues one commit for the whole batch instead of one per file.
    pub fn set_file_hashes_batch(&self, hashes: &[(String, String)]) -> Result<()> {
        if hashes.is_empty() {
            return Ok(());
        }

        let borrowed: Vec<(&str, &str)> = hashes
            .iter()
            .map(|(file_path, hash)| (file_path.as_str(), hash.as_str()))
            .collect();
        self.set_file_hashes_batch_borrowed(&borrowed)
    }

    pub(crate) fn set_file_hashes_batch_borrowed(&self, hashes: &[(&str, &str)]) -> Result<()> {
        if hashes.is_empty() {
            return Ok(());
        }

        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin file hash batch transaction")?;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO file_hashes (file_path, content_hash)
                     VALUES (?1, ?2)",
                )
                .context("failed to prepare batch file hash insert")?;
            for (file_path, hash) in hashes {
                stmt.execute(params![file_path, hash])
                    .with_context(|| format!("failed to set file hash: {file_path}"))?;
            }
        }
        tx.commit().context("failed to commit file hash batch")?;
        Ok(())
    }

    /// Store call-site references and type relations for many files in a
    /// single transaction.
    ///
    /// Equivalent to calling [`Self::insert_references`] and
    /// [`Self::insert_type_relations`] once per file, but issues one commit
    /// for the whole batch instead of up to two per file.
    pub fn insert_parse_artifacts_batch(
        &self,
        file_refs: &[(String, Vec<crate::parsing::references::RawReference>)],
        file_type_relations: &[(String, Vec<RawTypeRelation>)],
    ) -> Result<()> {
        if file_refs.is_empty() && file_type_relations.is_empty() {
            return Ok(());
        }

        let borrowed_refs: Vec<(&str, &[crate::parsing::references::RawReference])> = file_refs
            .iter()
            .map(|(file_path, refs)| (file_path.as_str(), refs.as_slice()))
            .collect();
        let borrowed_relations: Vec<(&str, &[RawTypeRelation])> = file_type_relations
            .iter()
            .map(|(file_path, relations)| (file_path.as_str(), relations.as_slice()))
            .collect();
        self.insert_parse_artifacts_batch_borrowed(&borrowed_refs, &borrowed_relations)
    }

    pub(crate) fn insert_parse_artifacts_batch_borrowed(
        &self,
        file_refs: &[(&str, &[crate::parsing::references::RawReference])],
        file_type_relations: &[(&str, &[RawTypeRelation])],
    ) -> Result<()> {
        if file_refs.is_empty() && file_type_relations.is_empty() {
            return Ok(());
        }

        let tx = self
            .conn
            .unchecked_transaction()
            .context("failed to begin parse artifacts batch transaction")?;
        {
            let mut ref_stmt = tx
                .prepare_cached(
                    "INSERT INTO [references] (file_path, line, callee, caller)
                     VALUES (?1, ?2, ?3, ?4)",
                )
                .context("failed to prepare batch reference insert")?;
            for &(file_path, refs) in file_refs {
                for r in refs {
                    ref_stmt
                        .execute(params![file_path, r.line, r.callee, r.caller])
                        .with_context(|| format!("failed to insert reference for {file_path}"))?;
                }
            }

            let mut relation_stmt = tx
                .prepare_cached(
                    "INSERT INTO type_relations (file_path, line, owner, target, kind)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .context("failed to prepare batch type relation insert")?;
            for &(file_path, relations) in file_type_relations {
                for relation in relations {
                    relation_stmt
                        .execute(params![
                            file_path,
                            relation.line,
                            relation.owner,
                            relation.target,
                            relation.kind.as_str(),
                        ])
                        .with_context(|| {
                            format!("failed to insert type relation for {file_path}")
                        })?;
                }
            }
        }
        tx.commit()
            .context("failed to commit parse artifacts batch")?;
        Ok(())
    }

    /// Delete all references for a given file.
    pub fn delete_references_by_file(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM [references] WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete references by file")?;
        Ok(())
    }

    /// Delete all explicit type relations for a given file.
    pub fn delete_type_relations_by_file(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM type_relations WHERE file_path = ?1",
                params![file_path],
            )
            .context("failed to delete type relations by file")?;
        Ok(())
    }

    /// Find all call sites that reference a given symbol name.
    pub fn find_callers(&self, symbol_name: &str) -> Result<Vec<CallerRef>> {
        let mut stmt = self
            .conn
            .prepare_cached(SQL_FIND_CALLERS)
            .context("failed to prepare callers query")?;
        let rows = stmt
            .query_map(params![symbol_name], |row| {
                Ok(CallerRef {
                    file_path: row.get(0)?,
                    line: row.get(1)?,
                    caller: row.get(2)?,
                })
            })
            .context("failed to query callers")?;
        collect_rows(rows)
    }

    /// Find explicit type relations that point at a given target symbol.
    pub fn find_type_relations(&self, symbol_name: &str) -> Result<Vec<TypeRelationRef>> {
        let mut stmt = self
            .conn
            .prepare_cached(SQL_FIND_TYPE_RELATIONS)
            .context("failed to prepare type relation query")?;
        let rows = stmt
            .query_map(params![symbol_name], |row| {
                Ok(TypeRelationRef {
                    file_path: row.get(0)?,
                    line: row.get(1)?,
                    owner: row.get(2)?,
                    target: row.get(3)?,
                    kind: TypeRelationKind::parse(&row.get::<_, String>(4)?).ok_or_else(|| {
                        rusqlite::Error::InvalidColumnType(
                            4,
                            "kind".to_string(),
                            rusqlite::types::Type::Text,
                        )
                    })?,
                })
            })
            .context("failed to query type relations")?;
        collect_rows(rows)
    }

    /// Find all symbols called by a given symbol name.
    pub fn find_callees(&self, symbol_name: &str) -> Result<Vec<CalleeRef>> {
        let mut stmt = self
            .conn
            .prepare_cached(SQL_FIND_CALLEES)
            .context("failed to prepare callees query")?;
        let rows = stmt
            .query_map(params![symbol_name], |row| {
                Ok(CalleeRef {
                    file_path: row.get(0)?,
                    line: row.get(1)?,
                    callee: row.get(2)?,
                })
            })
            .context("failed to query callees")?;
        collect_rows(rows)
    }

    /// Find defined symbols that have zero callers (potential dead code).
    ///
    /// Returns symbol names and their definition locations. Excludes common
    /// entry points (main, test functions, etc.).
    pub fn find_dead_symbols(&self) -> Result<Vec<DeadSymbol>> {
        let mut stmt = self
            .conn
            .prepare(SQL_FIND_DEAD_SYMBOLS)
            .context("failed to prepare dead symbols query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(DeadSymbol {
                    symbol_name: row.get(0)?,
                    file_path: row.get(1)?,
                    line: row.get(2)?,
                    symbol_type: row.get(3)?,
                })
            })
            .context("failed to query dead symbols")?;
        collect_rows(rows)
    }

    /// Get distinct file paths in the index.
    pub fn indexed_files(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT file_path FROM chunks ORDER BY file_path")
            .context("failed to prepare indexed files query")?;
        let rows = stmt
            .query_map([], |row| row.get(0))
            .context("failed to query indexed files")?;
        collect_rows(rows)
    }

    /// Count distinct files in the index.
    pub fn file_count(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(DISTINCT file_path) FROM chunks", [], |row| {
                row.get(0)
            })
            .context("failed to count files")?;
        Ok(count as u64)
    }

    /// Get language breakdown (language -> chunk count).
    pub fn language_stats(&self) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT language, COUNT(*) FROM chunks
                 GROUP BY language ORDER BY COUNT(*) DESC",
            )
            .context("failed to prepare language stats query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query language stats")?;
        Ok(collect_rows(rows)?
            .into_iter()
            .map(|(lang, count): (String, i64)| (lang, count as u64))
            .collect())
    }

    /// Get language breakdown by file count (language -> file count).
    pub fn language_file_counts(&self) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT language, COUNT(DISTINCT file_path) FROM chunks
                 GROUP BY language ORDER BY COUNT(DISTINCT file_path) DESC",
            )
            .context("failed to prepare language file counts query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query language file counts")?;
        Ok(collect_rows(rows)?
            .into_iter()
            .map(|(lang, count): (String, i64)| (lang, count as u64))
            .collect())
    }

    /// Collect persisted index health metrics from file-level states.
    pub fn index_health(&self) -> Result<IndexHealth> {
        let files_indexed: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM file_index_state WHERE status = 'indexed'",
                [],
                |row| row.get(0),
            )
            .context("failed to count indexed files for health")?;
        let files_with_tree_sitter_errors: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM file_index_state WHERE status = 'indexed' AND tree_has_error = 1",
                [],
                |row| row.get(0),
            )
            .context("failed to count tree-sitter errors for health")?;
        let files_using_tier0_fallback: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM file_index_state WHERE status = 'indexed' AND tier0_fallback = 1",
                [],
                |row| row.get(0),
            )
            .context("failed to count tier0 fallbacks for health")?;
        let files_with_parse_failures: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM file_index_state WHERE status = 'parse_error'",
                [],
                |row| row.get(0),
            )
            .context("failed to count parse failures for health")?;

        let mut stmt = self
            .conn
            .prepare(
                "SELECT
                    language,
                    SUM(CASE WHEN status = 'indexed' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'indexed' AND tree_has_error = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'indexed' AND tier0_fallback = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'parse_error' THEN 1 ELSE 0 END)
                 FROM file_index_state
                 GROUP BY language
                 ORDER BY (SUM(CASE WHEN status = 'indexed' THEN 1 ELSE 0 END) +
                           SUM(CASE WHEN status = 'parse_error' THEN 1 ELSE 0 END)) DESC,
                          language ASC",
            )
            .context("failed to prepare index health query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok(LanguageHealthStat {
                    language: row.get(0)?,
                    files_indexed: row.get::<_, i64>(1)? as u64,
                    files_with_tree_sitter_errors: row.get::<_, i64>(2)? as u64,
                    files_using_tier0_fallback: row.get::<_, i64>(3)? as u64,
                    files_with_parse_failures: row.get::<_, i64>(4)? as u64,
                })
            })
            .context("failed to execute index health query")?;

        let by_language = collect_rows(rows)?;

        Ok(IndexHealth {
            files_indexed: files_indexed as u64,
            files_with_tree_sitter_errors: files_with_tree_sitter_errors as u64,
            files_using_tier0_fallback: files_using_tier0_fallback as u64,
            files_with_parse_failures: files_with_parse_failures as u64,
            by_language,
        })
    }

    /// Get top-level directories with file counts.
    pub fn top_directories(&self, limit: usize) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT
                    CASE
                        WHEN instr(file_path, '/') > 0
                        THEN substr(file_path, 1, instr(file_path, '/') - 1)
                        ELSE '.'
                    END AS dir,
                    COUNT(DISTINCT file_path)
                 FROM chunks
                 GROUP BY dir
                 ORDER BY COUNT(DISTINCT file_path) DESC
                 LIMIT ?1",
            )
            .context("failed to prepare top directories query")?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query top directories")?;
        Ok(collect_rows(rows)?
            .into_iter()
            .map(|(dir, count): (String, i64)| (dir, count as u64))
            .collect())
    }

    /// Get total lines of code across all chunks.
    pub fn total_lines(&self) -> Result<u64> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(SUM(line_end - line_start + 1), 0) FROM chunks",
                [],
                |row| row.get(0),
            )
            .context("failed to count total lines")?;
        Ok(count as u64)
    }

    /// Get symbol type breakdown (symbol_type -> count).
    pub fn symbol_type_stats(&self) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT symbol_type, COUNT(*) FROM chunks
                 WHERE symbol_type IS NOT NULL
                 GROUP BY symbol_type ORDER BY COUNT(*) DESC",
            )
            .context("failed to prepare symbol type stats query")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query symbol type stats")?;
        Ok(collect_rows(rows)?
            .into_iter()
            .map(|(sym_type, count): (String, i64)| (sym_type, count as u64))
            .collect())
    }

    /// Get files with the most chunks (hotspots).
    pub fn hotspot_files(&self, limit: usize) -> Result<Vec<(String, u64)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT file_path, COUNT(*) FROM chunks
                 GROUP BY file_path ORDER BY COUNT(*) DESC LIMIT ?1",
            )
            .context("failed to prepare hotspot files query")?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .context("failed to query hotspot files")?;
        Ok(collect_rows(rows)?
            .into_iter()
            .map(|(path, count): (String, i64)| (path, count as u64))
            .collect())
    }

    /// Find likely entry point files (main.*, index.*, app.*, etc.).
    pub fn entry_points(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT DISTINCT file_path FROM chunks
                 WHERE file_path LIKE '%/main.%'
                    OR file_path LIKE 'main.%'
                    OR file_path LIKE '%/index.%'
                    OR file_path LIKE '%/app.%'
                    OR file_path LIKE 'app.%'
                    OR file_path LIKE '%/lib.%'
                    OR file_path LIKE 'lib.%'
                    OR file_path LIKE '%/mod.%'
                    OR file_path LIKE '%/server.%'
                 ORDER BY file_path",
            )
            .context("failed to prepare entry points query")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("failed to query entry points")?;
        collect_rows(rows)
    }
}

/// Convert a SQLite row into a Chunk.
fn row_to_chunk(row: &rusqlite::Row<'_>) -> Result<Chunk> {
    let id: String = row.get(0).context("missing id")?;
    let file_path: String = row.get(1).context("missing file_path")?;
    let line_start: u32 = row.get(2).context("missing line_start")?;
    let line_end: u32 = row.get(3).context("missing line_end")?;
    let content: String = row.get(4).context("missing content")?;
    let language_str: String = row.get(5).context("missing language")?;
    let symbol_type_str: Option<String> = row.get(6).context("missing symbol_type")?;
    let symbol_name: Option<String> = row.get(7).context("missing symbol_name")?;

    let language = parse_language(&language_str);
    let symbol_type = symbol_type_str.as_deref().map(parse_symbol_type);

    Ok(Chunk {
        id,
        file_path,
        line_start,
        line_end,
        content,
        language,
        symbol_type,
        symbol_name,
    })
}

/// Parse a language string back into the enum.
/// Delegates to `Language::from_str()` to stay in sync with the `Display` impl.
fn parse_language(s: &str) -> Language {
    s.parse::<Language>().unwrap_or(Language::Unknown)
}

/// Parse a symbol type string back into the enum.
fn parse_symbol_type(s: &str) -> SymbolType {
    match s {
        "function" => SymbolType::Function,
        "method" => SymbolType::Method,
        "class" => SymbolType::Class,
        "struct" => SymbolType::Struct,
        "enum" => SymbolType::Enum,
        "trait" => SymbolType::Trait,
        "interface" => SymbolType::Interface,
        "type_alias" => SymbolType::TypeAlias,
        "constant" => SymbolType::Constant,
        "variable" => SymbolType::Variable,
        "module" => SymbolType::Module,
        _ => SymbolType::Block,
    }
}

/// Collect fallible mapped rows into a Vec, attaching a read-failure context.
fn collect_rows<T>(
    rows: impl IntoIterator<Item = std::result::Result<T, rusqlite::Error>>,
) -> Result<Vec<T>> {
    rows.into_iter()
        .map(|row| row.context("failed to read row"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunks() -> Vec<Chunk> {
        vec![
            Chunk {
                id: "src/main.rs:0".to_string(),
                file_path: "src/main.rs".to_string(),
                line_start: 1,
                line_end: 5,
                content: "fn main() {\n    println!(\"hello\");\n}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("main".to_string()),
            },
            Chunk {
                id: "src/main.rs:1".to_string(),
                file_path: "src/main.rs".to_string(),
                line_start: 7,
                line_end: 12,
                content: "struct Config {\n    name: String,\n}".to_string(),
                language: Language::Rust,
                symbol_type: Some(SymbolType::Struct),
                symbol_name: Some("Config".to_string()),
            },
            Chunk {
                id: "src/lib.py:0".to_string(),
                file_path: "src/lib.py".to_string(),
                line_start: 1,
                line_end: 3,
                content: "def hello():\n    pass".to_string(),
                language: Language::Python,
                symbol_type: Some(SymbolType::Function),
                symbol_name: Some("hello".to_string()),
            },
        ]
    }

    #[test]
    fn insert_and_count() {
        let store = MetadataStore::open_in_memory().unwrap();
        let chunks = sample_chunks();
        store.insert_chunks(&chunks).unwrap();
        assert_eq!(store.chunk_count().unwrap(), 3);
    }

    /// The index expression has to match the query text exactly. If either
    /// side is edited without the other, SQLite silently falls back to a full
    /// scan and nothing fails — so assert on the plan, not just the results.
    #[test]
    fn case_insensitive_lookups_use_an_index_rather_than_scanning() {
        let store = MetadataStore::open_in_memory().unwrap();

        let plan_for = |sql: &str| -> String {
            let mut stmt = store
                .conn
                .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
                .unwrap();
            let rows: Vec<String> = stmt
                .query_map(params!["needle"], |row| row.get::<_, String>(3))
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            rows.join(" | ")
        };

        for (label, sql, expected_index) in [
            ("find_callers", SQL_FIND_CALLERS, "idx_refs_callee_lower"),
            ("find_callees", SQL_FIND_CALLEES, "idx_refs_caller_lower"),
            (
                "find_type_relations",
                SQL_FIND_TYPE_RELATIONS,
                "idx_type_relations_target_lower",
            ),
            (
                "get_chunks_by_symbol_name",
                SQL_CHUNKS_BY_SYMBOL_NAME,
                "idx_chunks_symbol_name_lower",
            ),
        ] {
            let plan = plan_for(sql);
            assert!(
                plan.contains(&format!("USING INDEX {expected_index}")),
                "{label} must use {expected_index}, but the plan was: {plan}"
            );
            assert!(
                plan.split(" | ").all(|detail| {
                    detail
                        .split_whitespace()
                        .next()
                        .is_none_or(|operation| operation != "SCAN")
                }),
                "{label} still scans: {plan}"
            );
        }
    }

    #[test]
    fn dead_symbol_lookup_seeks_the_reference_index() {
        // The correlated NOT EXISTS re-runs per candidate chunk, so a scan
        // here is O(chunks x references) rather than a constant overhead.
        let store = MetadataStore::open_in_memory().unwrap();
        let mut stmt = store
            .conn
            .prepare(&format!("EXPLAIN QUERY PLAN {SQL_FIND_DEAD_SYMBOLS}"))
            .unwrap();
        let plan: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let plan = plan.join(" | ");
        assert!(
            plan.contains("SEARCH r"),
            "the reference subquery must seek, not scan: {plan}"
        );
    }

    #[test]
    fn set_file_hashes_batch_persists_every_row_and_replaces_on_repeat() {
        let store = MetadataStore::open_in_memory().unwrap();
        store
            .set_file_hashes_batch(&[
                ("src/main.rs".to_string(), "hash-a".to_string()),
                ("src/lib.py".to_string(), "hash-b".to_string()),
            ])
            .unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().as_deref(),
            Some("hash-a")
        );
        assert_eq!(
            store.get_file_hash("src/lib.py").unwrap().as_deref(),
            Some("hash-b")
        );

        store
            .set_file_hashes_batch(&[("src/main.rs".to_string(), "hash-c".to_string())])
            .unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().as_deref(),
            Some("hash-c")
        );
        // Guards against a partial write taking out rows the batch never
        // mentioned: re-checking only src/main.rs would miss that.
        assert_eq!(
            store.get_file_hash("src/lib.py").unwrap().as_deref(),
            Some("hash-b")
        );
    }

    #[test]
    fn batch_writes_accept_empty_input() {
        // The batch methods early-return before opening a transaction. An
        // update run with nothing to write must not error.
        let store = MetadataStore::open_in_memory().unwrap();
        store.set_file_hashes_batch(&[]).unwrap();
        store.insert_parse_artifacts_batch(&[], &[]).unwrap();
        assert!(store.get_chunks_by_ids(&[]).unwrap().is_empty());
    }

    #[test]
    fn insert_parse_artifacts_batch_persists_both_kinds_across_files() {
        let store = MetadataStore::open_in_memory().unwrap();
        let refs = vec![
            (
                "src/main.rs".to_string(),
                vec![crate::parsing::references::RawReference {
                    callee: "helper".to_string(),
                    caller: Some("main".to_string()),
                    line: 3,
                }],
            ),
            (
                "src/lib.py".to_string(),
                vec![crate::parsing::references::RawReference {
                    callee: "helper".to_string(),
                    caller: None,
                    line: 9,
                }],
            ),
        ];
        let relations = vec![(
            "src/main.rs".to_string(),
            vec![RawTypeRelation {
                owner: "Config".to_string(),
                target: "Display".to_string(),
                line: 12,
                kind: TypeRelationKind::Conforms,
            }],
        )];
        store
            .insert_parse_artifacts_batch(&refs, &relations)
            .unwrap();

        // Both files' references land, not just the first. A count of 2 alone
        // would also pass if both rows were stored under one file path, so
        // assert the paths themselves.
        let mut caller_paths: Vec<String> = store
            .find_callers("helper")
            .unwrap()
            .into_iter()
            .map(|caller_ref| caller_ref.file_path)
            .collect();
        caller_paths.sort();
        assert_eq!(caller_paths, vec!["src/lib.py", "src/main.rs"]);
        assert_eq!(store.find_type_relations("Display").unwrap().len(), 1);
    }

    #[test]
    fn get_chunks_by_ids_returns_only_requested_rows() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let found = store
            .get_chunks_by_ids(&["src/main.rs:0".to_string(), "src/lib.py:0".to_string()])
            .unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found["src/main.rs:0"].symbol_name.as_deref(), Some("main"));
        assert!(!found.contains_key("src/main.rs:1"));

        // A missing id is skipped rather than erroring, which is what the
        // caller's "chunk metadata not found" branch relies on.
        let partial = store
            .get_chunks_by_ids(&["src/main.rs:0".to_string(), "does/not/exist:9".to_string()])
            .unwrap();
        assert_eq!(partial.len(), 1);
    }

    #[test]
    fn get_chunk_by_id() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunk = store.get_chunk("src/main.rs:0").unwrap().unwrap();
        assert_eq!(chunk.file_path, "src/main.rs");
        assert_eq!(chunk.symbol_name, Some("main".to_string()));
        assert_eq!(chunk.language, Language::Rust);
        assert_eq!(chunk.symbol_type, Some(SymbolType::Function));
        assert_eq!(chunk.line_start, 1);
        assert_eq!(chunk.line_end, 5);
    }

    #[test]
    fn get_nonexistent_chunk() {
        let store = MetadataStore::open_in_memory().unwrap();
        assert!(store.get_chunk("nonexistent").unwrap().is_none());
    }

    #[test]
    fn get_chunks_by_file() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store.get_chunks_by_file("src/main.rs").unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].id, "src/main.rs:0");
        assert_eq!(chunks[1].id, "src/main.rs:1");
    }

    #[test]
    fn get_chunks_by_symbol_name() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store.get_chunks_by_symbol_name("config").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");
    }

    #[test]
    fn get_chunks_by_symbol_name_case_sensitive() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store
            .get_chunks_by_symbol_name_case_sensitive("Config")
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");

        let lower = store
            .get_chunks_by_symbol_name_case_sensitive("config")
            .unwrap();
        assert!(lower.is_empty());
    }

    #[test]
    fn get_chunks_by_symbol_name_substring() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let chunks = store
            .get_chunks_by_symbol_name_substring("fig", 10)
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, "src/main.rs:1");
    }

    #[test]
    fn delete_chunks_by_file() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();
        assert_eq!(store.chunk_count().unwrap(), 3);

        let deleted = store.delete_chunks_by_file("src/main.rs").unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.chunk_count().unwrap(), 1);
    }

    #[test]
    fn clear_store() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();
        store.set_file_hash("src/main.rs", "abc123").unwrap();

        store.clear().unwrap();
        assert_eq!(store.chunk_count().unwrap(), 0);
        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());
    }

    #[test]
    fn file_hash_operations() {
        let store = MetadataStore::open_in_memory().unwrap();

        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());

        store.set_file_hash("src/main.rs", "hash1").unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().unwrap(),
            "hash1"
        );

        // Update hash
        store.set_file_hash("src/main.rs", "hash2").unwrap();
        assert_eq!(
            store.get_file_hash("src/main.rs").unwrap().unwrap(),
            "hash2"
        );

        store.delete_file_hash("src/main.rs").unwrap();
        assert!(store.get_file_hash("src/main.rs").unwrap().is_none());
    }

    #[test]
    fn indexed_files() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let files = store.indexed_files().unwrap();
        assert_eq!(files, vec!["src/lib.py", "src/main.rs"]);
    }

    #[test]
    fn language_stats() {
        let store = MetadataStore::open_in_memory().unwrap();
        store.insert_chunks(&sample_chunks()).unwrap();

        let stats = store.language_stats().unwrap();
        assert_eq!(stats.len(), 2);
        // Rust has 2, Python has 1
        assert_eq!(stats[0], ("rust".to_string(), 2));
        assert_eq!(stats[1], ("python".to_string(), 1));
    }

    #[test]
    fn type_relation_operations() {
        let store = MetadataStore::open_in_memory().unwrap();

        store
            .insert_type_relations(
                "src/types.ts",
                &[RawTypeRelation {
                    owner: "Repo".to_string(),
                    target: "Loader".to_string(),
                    line: 2,
                    kind: TypeRelationKind::Conforms,
                }],
            )
            .unwrap();

        let relations = store.find_type_relations("loader").unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].owner, "Repo");
        assert_eq!(relations[0].target, "Loader");
        assert_eq!(relations[0].kind, TypeRelationKind::Conforms);

        store.delete_type_relations_by_file("src/types.ts").unwrap();
        assert!(store.find_type_relations("Loader").unwrap().is_empty());
    }

    #[test]
    fn insert_replaces_existing() {
        let store = MetadataStore::open_in_memory().unwrap();
        let mut chunks = sample_chunks();
        store.insert_chunks(&chunks).unwrap();

        // Modify and re-insert
        chunks[0].content = "fn main() { /* updated */ }".to_string();
        store.insert_chunks(&chunks[..1]).unwrap();

        let chunk = store.get_chunk("src/main.rs:0").unwrap().unwrap();
        assert!(chunk.content.contains("updated"));
        assert_eq!(store.chunk_count().unwrap(), 3);
    }

    #[test]
    fn parse_language_roundtrip() {
        // Exhaustive list of ALL Language variants to catch future additions.
        let all_langs = vec![
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Python,
            Language::Go,
            Language::Java,
            Language::C,
            Language::Cpp,
            Language::Ruby,
            Language::Swift,
            Language::Kotlin,
            Language::Scala,
            Language::Zig,
            Language::Lua,
            Language::Bash,
            Language::CSharp,
            Language::Php,
            Language::Haskell,
            Language::Elixir,
            Language::Dart,
            Language::Sql,
            Language::Hcl,
            Language::Protobuf,
            // Tier 1B
            Language::Html,
            Language::Css,
            Language::Scss,
            Language::Vue,
            Language::GraphQl,
            Language::CMake,
            Language::Dockerfile,
            Language::Xml,
            // Tier 2A
            Language::ObjectiveC,
            Language::Perl,
            Language::Julia,
            Language::Nix,
            Language::OCaml,
            Language::Groovy,
            Language::Clojure,
            Language::CommonLisp,
            Language::Erlang,
            Language::FSharp,
            Language::Fortran,
            Language::PowerShell,
            Language::R,
            // Tier 2A batch 2
            Language::Matlab,
            Language::DLang,
            Language::Fish,
            Language::Zsh,
            Language::Luau,
            Language::Scheme,
            Language::Racket,
            Language::Elm,
            Language::Glsl,
            Language::Hlsl,
            // Tier 2B
            Language::Svelte,
            Language::Astro,
            Language::Makefile,
            Language::Ini,
            Language::Nginx,
            Language::Prisma,
            Language::Rst,
            // Tier 0
            Language::Toml,
            Language::Yaml,
            Language::Json,
            Language::Markdown,
            Language::Unknown,
        ];
        for lang in &all_langs {
            let s = lang.to_string();
            assert_eq!(parse_language(&s), *lang, "Failed roundtrip for {s}");
        }
    }

    #[test]
    fn parse_language_unknown_input() {
        assert_eq!(parse_language("nonexistent"), Language::Unknown);
        assert_eq!(parse_language(""), Language::Unknown);
    }
}
