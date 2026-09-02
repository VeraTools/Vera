//! Filter-during-scan eligibility map (issue #197).
//!
//! Lazily built per store + generation: chunk row (1-based sqlite rowid) ->
//! (path-id u32, language compact u16). Distinct path table (10-20k) is tested
//! against glob filters via `GlobMatcher` (memoized failing states) per query;
//! exact paths resolve directly via hash map; language resolves to compact id.
//! The map is validated against the existing `MetadataDbStamp` and
//! `VectorStore` generation; any mismatch invalidates or rebuilds it.
//! Representation: two parallel arrays (`path_ids`, `languages`) sized to
//! `max_rowid`, sentinel values for missing rows. Inside the flat SIMD scan
//! only eligible rows enter top-K, hydrating only filtered top-K.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};

use crate::types::{Language, SearchFilters};

static BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn eligibility_build_count() -> usize {
    BUILD_COUNT.load(Ordering::Relaxed)
}

pub fn reset_eligibility_build_count() {
    BUILD_COUNT.store(0, Ordering::Relaxed);
}

const SENTINEL_PATH_ID: u32 = u32::MAX;
const SENTINEL_LANGUAGE: u16 = u16::MAX;

/// Compact language map size sentinel; valid ids are 0..66.
pub(crate) fn language_to_compact(lang: Language) -> u16 {
    match lang {
        Language::Rust => 0,
        Language::TypeScript => 1,
        Language::JavaScript => 2,
        Language::Python => 3,
        Language::Go => 4,
        Language::Java => 5,
        Language::C => 6,
        Language::Cpp => 7,
        Language::Ruby => 8,
        Language::Swift => 9,
        Language::Kotlin => 10,
        Language::Scala => 11,
        Language::Zig => 12,
        Language::Lua => 13,
        Language::Bash => 14,
        Language::CSharp => 15,
        Language::Php => 16,
        Language::Haskell => 17,
        Language::Elixir => 18,
        Language::Dart => 19,
        Language::Sql => 20,
        Language::Hcl => 21,
        Language::Protobuf => 22,
        Language::Html => 23,
        Language::Css => 24,
        Language::Scss => 25,
        Language::Vue => 26,
        Language::GraphQl => 27,
        Language::CMake => 28,
        Language::Dockerfile => 29,
        Language::Xml => 30,
        Language::ObjectiveC => 31,
        Language::Perl => 32,
        Language::Julia => 33,
        Language::Nix => 34,
        Language::OCaml => 35,
        Language::Groovy => 36,
        Language::Clojure => 37,
        Language::CommonLisp => 38,
        Language::Erlang => 39,
        Language::FSharp => 40,
        Language::Fortran => 41,
        Language::PowerShell => 42,
        Language::R => 43,
        Language::Matlab => 44,
        Language::DLang => 45,
        Language::Fish => 46,
        Language::Zsh => 47,
        Language::Luau => 48,
        Language::Scheme => 49,
        Language::Racket => 50,
        Language::Elm => 51,
        Language::Glsl => 52,
        Language::Hlsl => 53,
        Language::Svelte => 54,
        Language::Astro => 55,
        Language::Makefile => 56,
        Language::Ini => 57,
        Language::Nginx => 58,
        Language::Prisma => 59,
        Language::Rst => 60,
        Language::Toml => 61,
        Language::Yaml => 62,
        Language::Json => 63,
        Language::Markdown => 64,
        Language::Unknown => 65,
    }
}

#[allow(dead_code)]
pub(crate) fn compact_to_language(id: u16) -> Language {
    match id {
        0 => Language::Rust,
        1 => Language::TypeScript,
        2 => Language::JavaScript,
        3 => Language::Python,
        4 => Language::Go,
        5 => Language::Java,
        6 => Language::C,
        7 => Language::Cpp,
        8 => Language::Ruby,
        9 => Language::Swift,
        10 => Language::Kotlin,
        11 => Language::Scala,
        12 => Language::Zig,
        13 => Language::Lua,
        14 => Language::Bash,
        15 => Language::CSharp,
        16 => Language::Php,
        17 => Language::Haskell,
        18 => Language::Elixir,
        19 => Language::Dart,
        20 => Language::Sql,
        21 => Language::Hcl,
        22 => Language::Protobuf,
        23 => Language::Html,
        24 => Language::Css,
        25 => Language::Scss,
        26 => Language::Vue,
        27 => Language::GraphQl,
        28 => Language::CMake,
        29 => Language::Dockerfile,
        30 => Language::Xml,
        31 => Language::ObjectiveC,
        32 => Language::Perl,
        33 => Language::Julia,
        34 => Language::Nix,
        35 => Language::OCaml,
        36 => Language::Groovy,
        37 => Language::Clojure,
        38 => Language::CommonLisp,
        39 => Language::Erlang,
        40 => Language::FSharp,
        41 => Language::Fortran,
        42 => Language::PowerShell,
        43 => Language::R,
        44 => Language::Matlab,
        45 => Language::DLang,
        46 => Language::Fish,
        47 => Language::Zsh,
        48 => Language::Luau,
        49 => Language::Scheme,
        50 => Language::Racket,
        51 => Language::Elm,
        52 => Language::Glsl,
        53 => Language::Hlsl,
        54 => Language::Svelte,
        55 => Language::Astro,
        56 => Language::Makefile,
        57 => Language::Ini,
        58 => Language::Nginx,
        59 => Language::Prisma,
        60 => Language::Rst,
        61 => Language::Toml,
        62 => Language::Yaml,
        63 => Language::Json,
        64 => Language::Markdown,
        65 => Language::Unknown,
        _ => Language::Unknown,
    }
}

pub(crate) fn parse_language_compact(s: &str) -> Option<u16> {
    // Language Display is lower-case; parsing is case-insensitive via eq_ignore_ascii_case in matches_file.
    // Try FromStr after lowercasing; it expects lower-case wire names.
    let lower = s.to_ascii_lowercase();
    match lower.parse::<Language>() {
        Ok(lang) => Some(language_to_compact(lang)),
        Err(_) => None,
    }
}

/// Whether the full filter set is map-evaluable (path globs, exact paths, language only).
/// Any other dimension (symbol_type, scope, include_generated) forces fallback.
pub fn is_map_evaluable(filters: &SearchFilters) -> bool {
    if filters.is_empty() {
        return false;
    }
    if filters.symbol_type.is_some()
        || filters.scope.is_some()
        || filters.include_generated.is_some()
    {
        return false;
    }
    // At least one of the supported dimensions must be present; otherwise an empty-but-not-empty
    // state would be weird (should have been caught by is_empty). But be explicit.
    !filters.path_glob.is_empty() || filters.exact_paths.is_some() || filters.language.is_some()
}

/// Eligibility map per store generation.
pub struct EligibilityMap {
    /// Distinct paths table, order is path-id assignment order (sorted for determinism).
    pub distinct_paths: Vec<String>,
    /// Map from normalized path string -> path-id. Normalized = replace '\\' with '/'.
    pub path_to_id: HashMap<String, u32>,
    /// Per-row path-id, length = max_rowid, index = rowid-1. SENTINEL for missing.
    pub path_ids: Vec<u32>,
    /// Per-row language compact, length = max_rowid, index = rowid-1. SENTINEL for missing.
    pub languages: Vec<u16>,
    /// Maximum rowid used to size arrays.
    pub max_rowid: i64,
}

impl EligibilityMap {
    /// Build from the two SQLite files. Returns error on any inconsistency -> caller falls back.
    pub fn build(metadata_path: &Path, vector_path: &Path) -> Result<Self> {
        BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
        // Open metadata read-only.
        let conn = Connection::open_with_flags(metadata_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| {
                format!(
                    "eligibility: failed to open metadata {}",
                    metadata_path.display()
                )
            })?;

        // ATTACH vector db as vecdb.
        let vector_path_str = vector_path
            .to_str()
            .context("eligibility: vector path is not valid unicode")?
            .replace('\'', "''");
        conn.execute_batch(&format!("ATTACH DATABASE '{vector_path_str}' AS vecdb"))
            .context("eligibility: failed to attach vector db")?;

        // Distinct paths.
        let distinct_paths: Vec<String> = {
            let mut stmt = conn
                .prepare("SELECT DISTINCT file_path FROM chunks ORDER BY file_path")
                .context("eligibility: failed to prepare distinct paths")?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .context("eligibility: failed to query distinct paths")?;
            let mut paths = Vec::new();
            for r in rows {
                paths.push(r.context("eligibility: failed to read distinct path")?);
            }
            paths
        };

        if distinct_paths.is_empty() {
            // Empty index edge: still build empty map with zero rows; but we need max_rowid.
            // Continue to allocate arrays (maybe zero).
        }

        let mut path_to_id: HashMap<String, u32> = HashMap::with_capacity(distinct_paths.len());
        for (idx, path) in distinct_paths.iter().enumerate() {
            let key = path.replace('\\', "/");
            // Note: if distinct table contains both 'a\b' and 'a/b' they'd collide; but indexing is consistent slash.
            path_to_id.insert(key, idx as u32);
        }

        // Max rowid.
        let max_rowid: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(rowid),0) FROM vecdb.chunk_id_map",
                [],
                |row| row.get(0),
            )
            .context("eligibility: failed to read max rowid")?;

        let size = if max_rowid < 0 { 0 } else { max_rowid as usize };
        let mut path_ids = vec![SENTINEL_PATH_ID; size];
        let mut languages = vec![SENTINEL_LANGUAGE; size];

        // If there are no rows, skip join.
        if size > 0 && !distinct_paths.is_empty() {
            let mut stmt = conn
                .prepare(
                    "SELECT vecdb.chunk_id_map.rowid, chunks.file_path, chunks.language \
                     FROM vecdb.chunk_id_map JOIN chunks ON chunks.id = vecdb.chunk_id_map.chunk_id \
                     ORDER BY vecdb.chunk_id_map.rowid",
                )
                .context("eligibility: failed to prepare join")?;
            let mut rows = stmt
                .query([])
                .context("eligibility: failed to query join")?;
            while let Some(row) = rows
                .next()
                .context("eligibility: failed to fetch join row")?
            {
                let rowid: i64 = row.get(0).context("eligibility: failed to read rowid")?;
                let file_path: String = row
                    .get(1)
                    .context("eligibility: failed to read file_path")?;
                let lang_str: String =
                    row.get(2).context("eligibility: failed to read language")?;
                if rowid <= 0 || rowid > max_rowid {
                    continue;
                }
                let idx = (rowid - 1) as usize;
                let key = file_path.replace('\\', "/");
                let pid = path_to_id.get(&key).copied().unwrap_or(SENTINEL_PATH_ID);
                let lang_compact = parse_language_compact(&lang_str)
                    .unwrap_or(language_to_compact(Language::Unknown));
                path_ids[idx] = pid;
                languages[idx] = lang_compact;
            }
        } else if size > 0 {
            // Distinct paths empty but rows exist? still need join but path map empty -> all sentinels remain.
            // Still run join to populate languages.
            let mut stmt = conn
                .prepare(
                    "SELECT vecdb.chunk_id_map.rowid, chunks.file_path, chunks.language \
                     FROM vecdb.chunk_id_map JOIN chunks ON chunks.id = vecdb.chunk_id_map.chunk_id \
                     ORDER BY vecdb.chunk_id_map.rowid",
                )
                .context("eligibility: failed to prepare join (empty distinct)")?;
            let mut rows = stmt
                .query([])
                .context("eligibility: failed to query join")?;
            while let Some(row) = rows
                .next()
                .context("eligibility: failed to fetch join row")?
            {
                let rowid: i64 = row.get(0).context("eligibility: rowid")?;
                let _file_path: String = row.get(1).context("eligibility: file_path")?;
                let lang_str: String = row.get(2).context("eligibility: language")?;
                if rowid <= 0 || rowid > max_rowid {
                    continue;
                }
                let idx = (rowid - 1) as usize;
                // path remains sentinel
                let lang_compact = parse_language_compact(&lang_str)
                    .unwrap_or(language_to_compact(Language::Unknown));
                languages[idx] = lang_compact;
            }
        }

        // DETACH to clean up (optional).
        let _ = conn.execute_batch("DETACH DATABASE vecdb");

        Ok(Self {
            distinct_paths,
            path_to_id,
            path_ids,
            languages,
            max_rowid,
        })
    }

    /// Verify the map is consistent with current max_rowid (e.g. after vector update).
    /// If size mismatches, caller should treat as stale and rebuild/fallback.
    pub fn is_consistent_with_max_rowid(&self, current_max_rowid: i64) -> bool {
        self.max_rowid == current_max_rowid
    }
}

#[derive(Debug, Clone)]
pub enum PathEligibility {
    All,
    Empty,
    Set(Vec<bool>),
}

pub struct QueryEligibility {
    pub path: PathEligibility,
    pub language: Option<u16>, // None => all, Some => specific compact; None + Path::All would be unfiltered but we never build for that.
    // For instrumentation: whether language was unknown -> empty
    pub language_empty: bool,
}

impl QueryEligibility {
    pub fn is_empty(&self) -> bool {
        matches!(self.path, PathEligibility::Empty) || self.language_empty
    }
}

/// Resolve a filter set against the map's distinct paths and language domain.
/// Returns Empty if nothing can match (honest empty without scan).
pub fn resolve_query_eligibility(
    map: &EligibilityMap,
    filters: &SearchFilters,
) -> Result<QueryEligibility> {
    // Path dimension.
    let path = if filters.path_glob.is_empty() && filters.exact_paths.is_none() {
        PathEligibility::All
    } else {
        // Compute glob-allowed set if needed.
        let glob_allowed: Option<Vec<bool>> = if !filters.path_glob.is_empty() {
            let mut allowed = vec![false; map.distinct_paths.len()];
            let mut any = false;
            for (idx, path) in map.distinct_paths.iter().enumerate() {
                for pattern in &filters.path_glob {
                    if crate::types::glob_matches(pattern, path) {
                        allowed[idx] = true;
                        any = true;
                        break;
                    }
                }
            }
            // If no glob matches and there is no exact filter, then empty.
            if !any && filters.exact_paths.is_none() {
                return Ok(QueryEligibility {
                    path: PathEligibility::Empty,
                    language: None,
                    language_empty: false,
                });
            }
            Some(allowed)
        } else {
            None
        };

        let exact_allowed: Option<Vec<bool>> = if let Some(ref exact) = filters.exact_paths {
            let mut allowed = vec![false; map.distinct_paths.len()];
            let mut any = false;
            for exact_path in exact.iter() {
                let key = exact_path.replace('\\', "/");
                // Note: we also need to handle normalization like leading ./ etc? Use same as path_to_id key: slash only.
                // But exact_paths equality in SearchFilters uses slash-only; so we match that.
                if let Some(&pid) = map.path_to_id.get(&key) {
                    allowed[pid as usize] = true;
                    any = true;
                } else {
                    // Also try with stripped leading ./ and trailing /
                    let stripped = key.strip_prefix("./").unwrap_or(&key).trim_end_matches('/');
                    if let Some(&pid2) = map.path_to_id.get(stripped) {
                        allowed[pid2 as usize] = true;
                        any = true;
                    }
                }
            }
            if !any {
                return Ok(QueryEligibility {
                    path: PathEligibility::Empty,
                    language: None,
                    language_empty: false,
                });
            }
            Some(allowed)
        } else {
            None
        };

        // Combine.
        match (glob_allowed, exact_allowed) {
            (None, None) => PathEligibility::All, // unreachable because we handled All case above
            (Some(g), None) => {
                if g.iter().all(|&v| !v) {
                    PathEligibility::Empty
                } else {
                    PathEligibility::Set(g)
                }
            }
            (None, Some(e)) => {
                if e.iter().all(|&v| !v) {
                    PathEligibility::Empty
                } else {
                    PathEligibility::Set(e)
                }
            }
            (Some(g), Some(e)) => {
                let mut inter = vec![false; map.distinct_paths.len()];
                let mut any = false;
                for i in 0..map.distinct_paths.len() {
                    if g[i] && e[i] {
                        inter[i] = true;
                        any = true;
                    }
                }
                if !any {
                    PathEligibility::Empty
                } else {
                    PathEligibility::Set(inter)
                }
            }
        }
    };

    // Language dimension.
    let (language, language_empty) = if let Some(ref lang_str) = filters.language {
        match parse_language_compact(lang_str) {
            Some(compact) => (Some(compact), false),
            None => {
                // Unknown language string => honest empty (no chunk can have that language)
                return Ok(QueryEligibility {
                    path,
                    language: None,
                    language_empty: true,
                });
            }
        }
    } else {
        (None, false)
    };

    // If path is empty, we can early return empty.
    if matches!(path, PathEligibility::Empty) || language_empty {
        return Ok(QueryEligibility {
            path: PathEligibility::Empty,
            language,
            language_empty: true,
        });
    }

    Ok(QueryEligibility {
        path,
        language,
        language_empty,
    })
}

/// Check whether a row (rowid) is eligible under the query eligibility.
/// `row_idx` is 0-based index = rowid-1. Caller validates bounds.
#[inline]
pub fn is_row_eligible(map: &EligibilityMap, query: &QueryEligibility, row_idx: usize) -> bool {
    if query.language_empty || matches!(query.path, PathEligibility::Empty) {
        return false;
    }
    // Path check.
    match &query.path {
        PathEligibility::All => {}
        PathEligibility::Empty => return false,
        PathEligibility::Set(allowed) => {
            let pid = map
                .path_ids
                .get(row_idx)
                .copied()
                .unwrap_or(SENTINEL_PATH_ID);
            if pid == SENTINEL_PATH_ID {
                return false;
            }
            if !allowed.get(pid as usize).copied().unwrap_or(false) {
                return false;
            }
        }
    }
    // Language check.
    if let Some(filter_lang) = query.language {
        let row_lang = map
            .languages
            .get(row_idx)
            .copied()
            .unwrap_or(SENTINEL_LANGUAGE);
        if row_lang == SENTINEL_LANGUAGE {
            return false;
        }
        if row_lang != filter_lang {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SearchFilters;
    use std::collections::HashSet;
    use std::sync::Arc;

    fn sample_map() -> EligibilityMap {
        // Distinct paths: src/a.rs, src/b.rs, tests/c.rs, src/video/player.rs
        let distinct = vec![
            "src/a.rs".to_string(),
            "src/b.rs".to_string(),
            "src/video/player.rs".to_string(),
            "tests/c.rs".to_string(),
        ];
        let mut path_to_id = HashMap::new();
        for (i, p) in distinct.iter().enumerate() {
            path_to_id.insert(p.replace('\\', "/"), i as u32);
        }
        // 4 rows
        let path_ids = vec![0, 1, 2, 3];
        let languages = vec![
            language_to_compact(Language::Rust),
            language_to_compact(Language::Python),
            language_to_compact(Language::Rust),
            language_to_compact(Language::Rust),
        ];
        EligibilityMap {
            distinct_paths: distinct,
            path_to_id,
            path_ids,
            languages,
            max_rowid: 4,
        }
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn is_map_evaluable_rejects_unsupported() {
        assert!(!is_map_evaluable(&SearchFilters::default()));
        let mut f = SearchFilters::default();
        f.language = Some("rust".to_string());
        assert!(is_map_evaluable(&f));
        f.scope = Some(crate::types::SearchScope::Source);
        assert!(!is_map_evaluable(&f));
        let mut f2 = SearchFilters {
            path_glob: vec!["src/**".to_string()],
            ..Default::default()
        };
        assert!(is_map_evaluable(&f2));
        f2.symbol_type = Some("function".to_string());
        assert!(!is_map_evaluable(&f2));
        let mut f3 = SearchFilters {
            path_glob: vec!["src/**".to_string()],
            ..Default::default()
        };
        f3.include_generated = Some(false);
        assert!(!is_map_evaluable(&f3));
        // Only path/language/exact are allowed, even if language missing but path present it's evaluable
        let f4 = SearchFilters {
            exact_paths: Some(Arc::new(HashSet::from(["src/a.rs".to_string()]))),
            ..Default::default()
        };
        assert!(is_map_evaluable(&f4));
    }

    #[test]
    fn resolve_path_only_glob() {
        let map = sample_map();
        let filters = SearchFilters {
            path_glob: vec!["src/**".to_string()],
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        assert!(!q.is_empty());
        match q.path {
            PathEligibility::Set(v) => {
                assert_eq!(v, vec![true, true, true, false]);
            }
            _ => panic!("expected Set"),
        }
        assert!(q.language.is_none());
    }

    #[test]
    fn resolve_exact_paths_direct() {
        let map = sample_map();
        let filters = SearchFilters {
            exact_paths: Some(Arc::new(HashSet::from(["src/a.rs".to_string()]))),
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        match q.path {
            PathEligibility::Set(v) => {
                assert_eq!(v, vec![true, false, false, false]);
            }
            _ => panic!("expected Set"),
        }
    }

    #[test]
    fn resolve_language_only() {
        let map = sample_map();
        let filters = SearchFilters {
            language: Some("rust".to_string()),
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        assert!(matches!(q.path, PathEligibility::All));
        assert_eq!(q.language, Some(language_to_compact(Language::Rust)));
    }

    #[test]
    fn resolve_combined_intersection() {
        let map = sample_map();
        let filters = SearchFilters {
            path_glob: vec!["src/**".to_string()],
            language: Some("python".to_string()),
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        // Path set includes src/a.rs, src/b.rs, src/video/player.rs
        // Language python only matches src/b.rs (idx 1)
        // So row eligible check: row idx 1 should be eligible, others not
        // But path eligibility itself is still src set; language filter restricts further per row
        match &q.path {
            PathEligibility::Set(v) => assert_eq!(v, &vec![true, true, true, false]),
            _ => panic!("expected set"),
        }
        assert_eq!(q.language, Some(language_to_compact(Language::Python)));
        // Check per row eligibility
        assert!(!is_row_eligible(&map, &q, 0)); // src/a.rs rust vs python filter
        assert!(is_row_eligible(&map, &q, 1)); // src/b.rs python
        assert!(!is_row_eligible(&map, &q, 2)); // src/video/player.rs rust
        assert!(!is_row_eligible(&map, &q, 3)); // tests/c.rs not in src glob
    }

    #[test]
    fn resolve_non_matching_glob_empty() {
        let map = sample_map();
        let filters = SearchFilters {
            path_glob: vec!["nonexistent/**".to_string()],
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        assert!(q.is_empty());
        assert!(matches!(q.path, PathEligibility::Empty));
    }

    #[test]
    fn resolve_unknown_language_empty() {
        let map = sample_map();
        let filters = SearchFilters {
            language: Some("nope_lang".to_string()),
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        assert!(q.is_empty());
        assert!(q.language_empty);
    }

    #[test]
    fn path_and_exact_intersection() {
        let map = sample_map();
        let filters = SearchFilters {
            path_glob: vec!["src/**".to_string()],
            exact_paths: Some(Arc::new(HashSet::from([
                "src/a.rs".to_string(),
                "tests/c.rs".to_string(),
            ]))),
            ..Default::default()
        };
        let q = resolve_query_eligibility(&map, &filters).unwrap();
        // Glob set: a,b,video ; exact set: a,tests/c ; intersection: a
        match q.path {
            PathEligibility::Set(v) => assert_eq!(v, vec![true, false, false, false]),
            _ => panic!("expected Set"),
        }
    }
}
