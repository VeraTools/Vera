//! Agent-oriented structural search intents built on top of the index.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Result, bail};
use cap_std::fs::Dir;
use regex::{Captures, Regex};
use tree_sitter::Parser;

use crate::corpus::{ContentClass, classify_content};
use crate::parsing::languages;
use crate::path_containment::canonical_project_root;
use crate::retrieval::apply_filters;
use crate::retrieval::file_scan::{
    allows_class, bounded_byte_snippet, language_for_path, smallest_symbol_chunk_for_line,
    sort_files_by_scan_priority, symbol_for_line,
};
use crate::storage::metadata::MetadataStore;
use crate::types::{Chunk, Language, SearchFilters, SearchResult, SearchScope};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static SYNTAX_FILTER_CREATIONS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralSearchKind {
    Definitions,
    EnvReads,
    RouteHandlers,
    SqlQueries,
    Implementations,
}

impl std::str::FromStr for StructuralSearchKind {
    type Err = ();

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "definitions" | "defs" => Ok(Self::Definitions),
            "env" | "env_reads" => Ok(Self::EnvReads),
            "routes" | "route_handlers" => Ok(Self::RouteHandlers),
            "sql" | "sql_queries" => Ok(Self::SqlQueries),
            "impls" | "implementations" => Ok(Self::Implementations),
            _ => Err(()),
        }
    }
}

pub fn search_structural(
    index_dir: &Path,
    kind: StructuralSearchKind,
    query: Option<&str>,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    if limit == 0 {
        bail!("limit must be greater than zero");
    }

    let metadata_path = index_dir.join("metadata.db");
    let store = MetadataStore::open(&metadata_path)?;
    let repo_root = canonical_project_root(index_dir)?;
    let root_dir = crate::discovery::open_root_dir(&repo_root)?;
    let max_file_size_bytes = super::configured_max_file_size_bytes(&store);
    let filters = structural_filters(filters);

    match kind {
        StructuralSearchKind::Definitions => {
            let symbol = required_query(kind, query)?;
            search_definitions(&store, symbol, limit, &filters)
        }
        StructuralSearchKind::EnvReads => search_env_reads(
            &root_dir,
            &store,
            max_file_size_bytes,
            query,
            limit,
            &filters,
        ),
        StructuralSearchKind::RouteHandlers => {
            reject_query(kind, query)?;
            search_route_handlers(&root_dir, &store, max_file_size_bytes, limit, &filters)
        }
        StructuralSearchKind::SqlQueries => {
            reject_query(kind, query)?;
            search_sql_queries(&root_dir, &store, max_file_size_bytes, limit, &filters)
        }
        StructuralSearchKind::Implementations => {
            let target = required_query(kind, query)?;
            search_implementations(index_dir, target, limit, &filters)
        }
    }
}

fn structural_filters(filters: &SearchFilters) -> SearchFilters {
    let mut filters = filters.clone();
    if filters.scope.is_none() {
        filters.scope = Some(SearchScope::Source);
    }
    filters
}

fn required_query(kind: StructuralSearchKind, query: Option<&str>) -> Result<&str> {
    non_blank(query).ok_or_else(|| anyhow::anyhow!("{} requires a query", kind_label(kind)))
}

fn non_blank(query: Option<&str>) -> Option<&str> {
    query.map(str::trim).filter(|value| !value.is_empty())
}

/// `ROUTE_PATTERNS` and `SQL_PATTERNS` carry no capture group, so unlike `ENV_PATTERNS`
/// there is no term-bearing entity for a query to narrow against. Dropping the argument
/// instead of rejecting it makes an unfiltered result set read as a match failure.
fn reject_query(kind: StructuralSearchKind, query: Option<&str>) -> Result<()> {
    match non_blank(query) {
        Some(value) => bail!(
            "{} accepts no query term; got {value:?}. Narrow with path, language, or scope filters instead.",
            kind_label(kind)
        ),
        None => Ok(()),
    }
}

fn kind_label(kind: StructuralSearchKind) -> &'static str {
    match kind {
        StructuralSearchKind::Definitions => "definitions",
        StructuralSearchKind::EnvReads => "env reads",
        StructuralSearchKind::RouteHandlers => "route handlers",
        StructuralSearchKind::SqlQueries => "SQL queries",
        StructuralSearchKind::Implementations => "implementations",
    }
}

fn search_definitions(
    store: &MetadataStore,
    symbol: &str,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let chunks = {
        let exact = store.get_chunks_by_symbol_name_case_sensitive(symbol)?;
        if exact.is_empty() {
            store.get_chunks_by_symbol_name(symbol)?
        } else {
            exact
        }
    };

    let results = chunks
        .into_iter()
        .map(|chunk| chunk.into_search_result(1.0))
        .collect();

    Ok(apply_filters(results, filters, limit))
}

fn search_env_reads(
    root_dir: &Dir,
    store: &MetadataStore,
    max_file_size_bytes: u64,
    query: Option<&str>,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    let target = non_blank(query);
    search_regex_intent(
        root_dir,
        store,
        max_file_size_bytes,
        limit,
        filters,
        true,
        |_, language, content| {
            let Some(patterns) = patterns_for_language(&ENV_PATTERNS, language) else {
                return Ok(Vec::new());
            };
            let mut results = Vec::new();
            for pattern in patterns {
                for captures in pattern.captures_iter(content) {
                    let Some(full) = captures.get(0) else {
                        continue;
                    };
                    let Some(found_name) = first_capture(&captures) else {
                        continue;
                    };
                    if let Some(target) = target
                        && !found_name.eq_ignore_ascii_case(target)
                    {
                        continue;
                    }
                    results.push(StructuralMatch {
                        start_byte: full.start(),
                        end_byte: full.end(),
                    });
                }
            }
            Ok(results)
        },
    )
}

fn search_route_handlers(
    root_dir: &Dir,
    store: &MetadataStore,
    max_file_size_bytes: u64,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    search_regex_intent(
        root_dir,
        store,
        max_file_size_bytes,
        limit,
        filters,
        true,
        |_, language, content| {
            let Some(patterns) = patterns_for_language(&ROUTE_PATTERNS, language) else {
                return Ok(Vec::new());
            };
            Ok(patterns
                .iter()
                .flat_map(|pattern| pattern.find_iter(content))
                .map(|found| StructuralMatch {
                    start_byte: found.start(),
                    end_byte: found.end(),
                })
                .collect())
        },
    )
}

fn search_sql_queries(
    root_dir: &Dir,
    store: &MetadataStore,
    max_file_size_bytes: u64,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    search_regex_intent(
        root_dir,
        store,
        max_file_size_bytes,
        limit,
        filters,
        true,
        |_, language, content| {
            let Some(patterns) = patterns_for_language(&SQL_PATTERNS, language) else {
                return Ok(Vec::new());
            };
            Ok(patterns
                .iter()
                .flat_map(|pattern| pattern.find_iter(content))
                .map(|found| StructuralMatch {
                    start_byte: found.start(),
                    end_byte: found.end(),
                })
                .collect())
        },
    )
}

fn search_implementations(
    index_dir: &Path,
    target: &str,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<SearchResult>> {
    super::type_relations::search_explicit_implementations(index_dir, target, limit, filters)
}

fn search_regex_intent<F>(
    root_dir: &Dir,
    store: &MetadataStore,
    max_file_size_bytes: u64,
    limit: usize,
    filters: &SearchFilters,
    prefer_chunk: bool,
    mut collect: F,
) -> Result<Vec<SearchResult>>
where
    F: FnMut(&str, Language, &str) -> Result<Vec<StructuralMatch>>,
{
    let mut files = store.indexed_files()?;
    sort_files_by_scan_priority(&mut files, filters);

    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for file_rel in files {
        if results.len() >= limit {
            break;
        }

        let language = language_for_path(&file_rel);
        if !filters.matches_file(&file_rel, language) {
            continue;
        }

        let content = match crate::discovery::read_source_lossy_capped(
            root_dir,
            Path::new(&file_rel),
            max_file_size_bytes,
        ) {
            Ok(content) => content,
            Err(e) => {
                tracing::debug!("skipping {}: {e}", file_rel);
                continue;
            }
        };
        let class = classify_content(&file_rel, language, &content);
        if !allows_class(filters, class) {
            continue;
        }
        if matches!(filters.include_generated, Some(false))
            && matches!(class, ContentClass::Generated)
        {
            continue;
        }

        let candidates = collect(&file_rel, language, &content)?;
        if candidates.is_empty() {
            continue;
        }

        let chunks = store.get_chunks_by_file(&file_rel)?;
        let syntax_filter = SyntaxFilter::new(language, &file_rel, &content);
        for candidate in candidates {
            if results.len() >= limit {
                break;
            }
            if !syntax_filter.allows(candidate.start_byte, candidate.end_byte) {
                continue;
            }
            let result = result_for_match(
                &file_rel,
                language,
                &content,
                candidate.start_byte,
                candidate.end_byte,
                &chunks,
                prefer_chunk,
            );
            if !filters.matches_symbol_type(result.symbol_type) {
                continue;
            }
            let key = format!(
                "{}:{}:{}",
                result.file_path, result.line_start, result.line_end
            );
            if seen.insert(key) {
                results.push(result);
            }
        }
    }

    Ok(results)
}

fn result_for_match(
    file_path: &str,
    language: Language,
    content: &str,
    start_byte: usize,
    end_byte: usize,
    chunks: &[Chunk],
    prefer_chunk: bool,
) -> SearchResult {
    let line = crate::retrieval::file_scan::byte_to_line(content, start_byte);
    if prefer_chunk
        && let Some(chunk) = smallest_symbol_chunk_for_line(chunks, line)
            .filter(|chunk| chunk.line_end.saturating_sub(chunk.line_start) <= 80)
    {
        return SearchResult {
            file_path: file_path.to_string(),
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            content: chunk.content.clone(),
            language,
            score: 1.0,
            symbol_name: chunk.symbol_name.clone(),
            symbol_type: chunk.symbol_type,
        };
    }

    let (snippet, line_start, line_end) = bounded_byte_snippet(content, start_byte, end_byte, 220);
    let (symbol_name, symbol_type) = symbol_for_line(Some(chunks), line);
    SearchResult {
        file_path: file_path.to_string(),
        line_start,
        line_end,
        content: snippet,
        language,
        score: 1.0,
        symbol_name,
        symbol_type,
    }
}

fn first_capture<'a>(captures: &'a Captures<'a>) -> Option<&'a str> {
    captures
        .iter()
        .skip(1)
        .flatten()
        .next()
        .map(|value| value.as_str())
}

type PatternSets = Vec<(Vec<Language>, Vec<Regex>)>;

#[derive(Debug, Clone, Copy)]
struct StructuralMatch {
    start_byte: usize,
    end_byte: usize,
}

struct SyntaxFilter(Option<tree_sitter::Tree>);

impl SyntaxFilter {
    fn new(language: Language, file_path: &str, content: &str) -> Self {
        #[cfg(test)]
        SYNTAX_FILTER_CREATIONS.with(|count| count.set(count.get() + 1));

        let tree =
            languages::tree_sitter_grammar_for_path(language, file_path).and_then(|grammar| {
                let mut parser = Parser::new();
                parser.set_language(&grammar).ok()?;
                parser.parse(content, None)
            });
        Self(tree)
    }

    fn allows(&self, start_byte: usize, end_byte: usize) -> bool {
        let Some(tree) = &self.0 else {
            return true;
        };
        let Some(mut node) = tree
            .root_node()
            .descendant_for_byte_range(start_byte, end_byte.max(start_byte + 1))
        else {
            return true;
        };
        loop {
            if is_ignorable_syntax_kind(node.kind()) {
                return false;
            }
            let Some(parent) = node.parent() else {
                return true;
            };
            node = parent;
        }
    }
}

fn is_ignorable_syntax_kind(kind: &str) -> bool {
    kind.contains("comment")
        || kind.contains("string")
        || matches!(
            kind,
            "template_string"
                | "raw_string_literal"
                | "interpreted_string_literal"
                | "char_literal"
                | "rune_literal"
                | "heredoc_body"
        )
}

fn patterns_for_language(pattern_sets: &PatternSets, language: Language) -> Option<&[Regex]> {
    pattern_sets
        .iter()
        .find(|(languages, _)| languages.contains(&language))
        .map(|(_, patterns)| patterns.as_slice())
}

static ENV_PATTERNS: LazyLock<PatternSets> = LazyLock::new(|| {
    vec![
        (
            vec![Language::JavaScript, Language::TypeScript],
            compile_patterns(&[
                r#"process\.env(?:\.([A-Za-z_][A-Za-z0-9_]*)|\[\s*["']([^"'\\]+)["']\s*\])"#,
                r#"(?:import\.meta|Deno)\.env\.get\(\s*["']([^"'\\]+)["']\s*\)"#,
            ]),
        ),
        (
            vec![Language::Python],
            compile_patterns(&[
                r#"os\.getenv\(\s*["']([^"'\\]+)["']\s*\)"#,
                r#"os\.environ(?:\.get\(\s*["']([^"'\\]+)["']\s*\)|\[\s*["']([^"'\\]+)["']\s*\])"#,
            ]),
        ),
        (
            vec![Language::Rust],
            compile_patterns(&[
                r#"(?:std::)?env::(?:var|var_os)\(\s*"([^"\\]+)""#,
                r#"option_env!\(\s*"([^"\\]+)""#,
            ]),
        ),
        (
            vec![Language::Go],
            compile_patterns(&[r#"os\.(?:Getenv|LookupEnv)\(\s*"([^"\\]+)""#]),
        ),
        (
            vec![Language::Java],
            compile_patterns(&[r#"System\.getenv\(\s*"([^"\\]+)""#]),
        ),
        (
            vec![Language::CSharp],
            compile_patterns(&[r#"Environment\.GetEnvironmentVariable\(\s*"([^"\\]+)""#]),
        ),
    ]
});

static ROUTE_PATTERNS: LazyLock<PatternSets> = LazyLock::new(|| {
    vec![
        (
            vec![Language::JavaScript, Language::TypeScript],
            compile_patterns(&[
                r#"\.(?:get|post|put|patch|delete|all|use)\s*\(\s*["'`][^"'`]+["'`]"#,
                r#"\.route\(\s*["'`][^"'`]+["'`]\s*\)"#,
            ]),
        ),
        (
            vec![Language::Python],
            compile_patterns(&[
                r#"@(?:\w+\.)?(?:get|post|put|patch|delete|route)\(\s*["'][^"'\\]+["']"#,
                r#"\.add_api_route\(\s*["'][^"'\\]+["']"#,
            ]),
        ),
        (
            vec![Language::Rust],
            compile_patterns(&[
                r#"#\[(?:get|post|put|patch|delete|route)\(\s*"[^"\\]+""#,
                r#"\.route\(\s*"[^"\\]+""#,
            ]),
        ),
        (
            vec![Language::Go],
            compile_patterns(&[
                r#"\.(?:GET|POST|PUT|PATCH|DELETE|HandleFunc|Handle)\(\s*"[^"\\]+""#,
                r#"http\.HandleFunc\(\s*"[^"\\]+""#,
            ]),
        ),
        (
            vec![Language::Java],
            compile_patterns(&[
                r#"@(?:GetMapping|PostMapping|PutMapping|PatchMapping|DeleteMapping|RequestMapping)\(\s*(?:value\s*=\s*)?"[^"\\]+""#,
            ]),
        ),
        (
            vec![Language::CSharp],
            compile_patterns(&[
                r#"Map(?:Get|Post|Put|Patch|Delete)\(\s*"[^"\\]+""#,
                r#"\[(?:HttpGet|HttpPost|HttpPut|HttpPatch|HttpDelete)(?:\(\s*"[^"\\]+"\s*\))?\]"#,
            ]),
        ),
    ]
});

static SQL_PATTERNS: LazyLock<PatternSets> = LazyLock::new(|| {
    vec![
        (
            vec![Language::JavaScript, Language::TypeScript],
            compile_patterns(&[
                r#"\b(?:db|pool|client|conn|connection|trx|tx|prisma|sequelize|knex)\s*\.\s*(?:query|execute|queryRaw|queryUnsafe|executeRaw|raw)\s*\("#,
                r#"\bsql\s*`"#,
            ]),
        ),
        (
            vec![Language::Python],
            compile_patterns(&[
                r#"\b(?:cursor|conn|connection|db|session|engine)\s*\.\s*(?:execute|executemany|exec_driver_sql)\s*\("#,
            ]),
        ),
        (
            vec![Language::Rust],
            compile_patterns(&[
                r#"\b(?:sqlx|diesel)::(?:query|query_as|query_scalar|sql_query)\s*!?\s*\("#,
                r#"\b(?:conn|pool|tx|db)\s*\.\s*(?:execute|fetch_one|fetch_all|fetch_optional|query)\s*\("#,
            ]),
        ),
        (
            vec![Language::Go],
            compile_patterns(&[
                r#"\b(?:db|tx|conn)\s*\.\s*(?:Query|QueryContext|Exec|ExecContext|Prepare)\s*\("#,
            ]),
        ),
        (
            vec![Language::Java],
            compile_patterns(&[
                r#"\b(?:statement|preparedStatement|entityManager|query)\s*\.\s*(?:executeQuery|executeUpdate|prepareStatement|createQuery|createNativeQuery)\s*\("#,
            ]),
        ),
        (
            vec![Language::CSharp],
            compile_patterns(&[
                r#"\b(?:db|context|command|connection)\s*\.\s*(?:ExecuteReader|ExecuteNonQuery|ExecuteSqlRaw|ExecuteSqlInterpolated|FromSqlRaw|FromSqlInterpolated|Query|QueryAsync)\s*\("#,
            ]),
        ),
    ]
});

fn compile_patterns(patterns: &[&str]) -> Vec<Regex> {
    patterns
        .iter()
        .map(|pattern| Regex::new(pattern).expect("structural regex should compile"))
        .collect()
}

pub(crate) fn normalize_impl_target(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('&')
        .split('<')
        .next()
        .unwrap_or("")
        .split_whitespace()
        .last()
        .unwrap_or("")
        .rsplit([':', '.'])
        .next()
        .unwrap_or("")
        .trim_matches(|ch: char| ch == '{' || ch == ')' || ch == '(')
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VeraConfig;
    use crate::embedding::test_helpers::MockProvider;
    use crate::indexing::index_repository;

    async fn index_repo(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for (path, content) in files {
            let abs = dir.path().join(path);
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(abs, content).unwrap();
        }
        let provider = MockProvider::new(8);
        let config = VeraConfig::default();
        index_repository(dir.path(), &provider, &config, "mock-model")
            .await
            .unwrap();
        dir
    }

    #[tokio::test]
    async fn definitions_find_symbol_chunks() {
        let dir = index_repo(&[("src/main.rs", "fn parse_config() {}\nfn other() {}\n")]).await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::Definitions,
            Some("parse_config"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol_name.as_deref(), Some("parse_config"));
    }

    /// Structural search over a JSX-bearing `.tsx` file: the real env read is
    /// found and attributed, the comment and JSX-attribute decoys are not.
    ///
    /// This characterises behaviour rather than guarding the grammar switch.
    /// It passes under both grammars, which is the point: tree-sitter still
    /// lexes comments and strings inside an error region, and still recognises
    /// the enclosing `function_declaration`, so swapping `typescript` for `tsx`
    /// leaves structural results unchanged. Nothing covered `.tsx` structural
    /// search before, so the coverage is new either way.
    #[tokio::test]
    async fn structural_search_works_in_jsx_bearing_tsx() {
        let dir = index_repo(&[(
            "src/Panel.tsx",
            r#"export function Panel() {
  // const decoy = process.env.FAKE_URL;
  const real = process.env.API_URL;
  return <div title="process.env.FAKE_URL">{real}</div>;
}
"#,
        )])
        .await;
        let index_dir = crate::indexing::index_dir(dir.path());

        let real = search_structural(
            &index_dir,
            StructuralSearchKind::EnvReads,
            Some("API_URL"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert_eq!(real.len(), 1, "real env read should be found: {real:?}");
        assert_eq!(real[0].file_path, "src/Panel.tsx");
        assert_eq!(
            real[0].symbol_name.as_deref(),
            Some("Panel"),
            "match should be attributed to the enclosing component"
        );

        let decoy = search_structural(
            &index_dir,
            StructuralSearchKind::EnvReads,
            Some("FAKE_URL"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert!(
            decoy.is_empty(),
            "comment and JSX attribute string should be filtered: {decoy:?}"
        );
    }

    #[tokio::test]
    async fn definitions_find_const_assigned_functions() {
        let dir = index_repo(&[(
            "src/shapes.ts",
            "export function declaredFn(a: number): number { return a; }\n\
             export const arrowFn = (a: number): number => a;\n\
             export const MyButton: React.FC = () => null;\n",
        )])
        .await;
        let index_dir = crate::indexing::index_dir(dir.path());

        for symbol in ["declaredFn", "arrowFn", "MyButton"] {
            let results = search_structural(
                &index_dir,
                StructuralSearchKind::Definitions,
                Some(symbol),
                10,
                &SearchFilters::default(),
            )
            .unwrap();
            assert_eq!(results.len(), 1, "no definition found for {symbol}");
            assert_eq!(results[0].symbol_name.as_deref(), Some(symbol));
        }
    }

    #[tokio::test]
    async fn env_reads_find_common_patterns() {
        let dir = index_repo(&[
            ("src/app.ts", "const db = process.env.DATABASE_URL;\n"),
            (
                "server.py",
                "import os\nvalue = os.environ.get('DATABASE_URL')\n",
            ),
            ("src/main.rs", "let v = std::env::var(\"DATABASE_URL\");\n"),
        ])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::EnvReads,
            Some("DATABASE_URL"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn route_handlers_find_common_patterns() {
        let dir = index_repo(&[
            ("src/router.ts", "router.get('/users', handler)\n"),
            ("app.py", "@app.post('/login')\ndef login():\n    pass\n"),
        ])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::RouteHandlers,
            None,
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn sql_queries_find_execution_sites() {
        let dir = index_repo(&[
            (
                "db.py",
                "def load(cursor):\n    cursor.execute('SELECT * FROM users')\n",
            ),
            ("src/main.rs", "let query = sqlx::query(\"SELECT 1\");\n"),
        ])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::SqlQueries,
            None,
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn implementations_find_rust_trait_impls() {
        let dir = index_repo(&[
            (
                "src/main.rs",
                "impl std::fmt::Display for User {\n    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { todo!() }\n}\n",
            ),
        ])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::Implementations,
            Some("Display"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .content
                .contains("impl std::fmt::Display for User")
        );
        assert_eq!(results[0].symbol_name.as_deref(), Some("User"));
        assert!(results[0].symbol_type.is_none());
    }

    #[tokio::test]
    async fn implementations_find_explicit_relations_across_languages() {
        let dir = index_repo(&[
            (
                "src/types.ts",
                "interface Loader {}\nclass Repo extends BaseRepo implements Loader {\n    run() {}\n}\n",
            ),
            (
                "src/Worker.java",
                "interface Loader {}\nfinal class Worker implements Loader {\n}\n",
            ),
            (
                "src/runner.cs",
                "public interface ILoader {}\npublic class Runner : BackgroundService, ILoader {\n}\n",
            ),
        ])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::Implementations,
            Some("Loader"),
            20,
            &SearchFilters::default(),
        )
        .unwrap();
        assert_eq!(results.len(), 2, "unexpected Loader matches: {results:?}");
        assert!(
            results
                .iter()
                .any(|result| result.file_path == "src/types.ts")
        );
        assert!(
            results
                .iter()
                .any(|result| result.file_path == "src/Worker.java")
        );
        assert!(results.iter().all(|result| result.symbol_name.is_some()));

        let runner_results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::Implementations,
            Some("ILoader"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert_eq!(runner_results.len(), 1);
        assert_eq!(runner_results[0].symbol_name.as_deref(), Some("Runner"));
        assert!(runner_results[0].content.contains("class Runner"));
    }

    #[tokio::test]
    async fn implementations_ignore_implicit_languages() {
        let dir = index_repo(&[(
            "src/repo.go",
            "type Loader interface {\n    Load() error\n}\n\ntype Repo struct {}\n\nfunc (Repo) Load() error { return nil }\n",
        )])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::Implementations,
            Some("Loader"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert!(
            results.is_empty(),
            "unexpected implicit impl matches: {results:?}"
        );
    }

    #[tokio::test]
    async fn structural_defaults_to_source_scope() {
        let dir = index_repo(&[
            ("src/app.ts", "const db = process.env.DATABASE_URL;\n"),
            (
                "docs/guide.ts",
                "export const example = process.env.DATABASE_URL;\n",
            ),
        ])
        .await;

        let default_results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::EnvReads,
            Some("DATABASE_URL"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();
        assert_eq!(default_results.len(), 1);
        assert_eq!(default_results[0].file_path, "src/app.ts");

        let docs_results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::EnvReads,
            Some("DATABASE_URL"),
            10,
            &SearchFilters {
                scope: Some(SearchScope::Docs),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(docs_results.len(), 1);
        assert_eq!(docs_results[0].file_path, "docs/guide.ts");
    }

    #[tokio::test]
    async fn sql_ignores_strings_and_comments() {
        let dir = index_repo(&[(
            "src/fixture.py",
            r#"def fake():
    # cursor.execute("SELECT * FROM users")
    sample = "cursor.execute('SELECT * FROM users')"
    return sample
"#,
        )])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::SqlQueries,
            None,
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert!(results.is_empty(), "unexpected SQL matches: {results:?}");
    }

    #[tokio::test]
    async fn routes_ignore_comment_examples() {
        let dir = index_repo(&[(
            "src/router.ts",
            r#"export function explain() {
    // router.get('/fake', handler)
    const example = "router.get('/fake', handler)";
    return example;
}
"#,
        )])
        .await;

        let results = search_structural(
            &crate::indexing::index_dir(dir.path()),
            StructuralSearchKind::RouteHandlers,
            None,
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        assert!(results.is_empty(), "unexpected route matches: {results:?}");
    }

    /// Routes and SQL cannot narrow by term, so a supplied query has to fail loudly.
    /// Dropping it returned the same unfiltered set the no-query call returns, which a
    /// caller reads as "no match for my query".
    #[tokio::test]
    async fn routes_and_sql_reject_a_query_they_cannot_honour() {
        let dir = index_repo(&[
            ("src/router.ts", "router.get('/users', handler)\n"),
            (
                "db.py",
                "def load(cursor):\n    cursor.execute('SELECT * FROM users')\n",
            ),
        ])
        .await;
        let index_dir = crate::indexing::index_dir(dir.path());

        for (kind, label) in [
            (StructuralSearchKind::RouteHandlers, "route handlers"),
            (StructuralSearchKind::SqlQueries, "SQL queries"),
        ] {
            let unfiltered =
                search_structural(&index_dir, kind, None, 10, &SearchFilters::default()).unwrap();
            assert_eq!(
                unfiltered.len(),
                1,
                "{label} fixture must produce the hit a dropped query would return: {unfiltered:?}"
            );

            let error = search_structural(
                &index_dir,
                kind,
                Some("find_user_by_email"),
                10,
                &SearchFilters::default(),
            )
            .expect_err(&format!("{label} accepted a query it cannot honour"))
            .to_string();
            assert!(
                error.contains(label) && error.contains("find_user_by_email"),
                "error must name the kind and the rejected term: {error}"
            );

            for blank in ["", "   "] {
                let results =
                    search_structural(&index_dir, kind, Some(blank), 10, &SearchFilters::default())
                        .unwrap();
                assert_eq!(
                    results.len(),
                    1,
                    "{label} must treat a blank query as absent: {results:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn unmatched_intent_skips_syntax_filter_parsing() {
        let dir = index_repo(&[("src/app.ts", "const value = 1;\n")]).await;
        let index_dir = crate::indexing::index_dir(dir.path());
        let before = SYNTAX_FILTER_CREATIONS.with(|count| count.get());

        let results = search_structural(
            &index_dir,
            StructuralSearchKind::EnvReads,
            Some("MISSING_ENV_NAME"),
            10,
            &SearchFilters::default(),
        )
        .unwrap();

        let after = SYNTAX_FILTER_CREATIONS.with(|count| count.get());
        assert!(results.is_empty());
        assert_eq!(after, before, "unmatched files must not be parsed");
    }
}
