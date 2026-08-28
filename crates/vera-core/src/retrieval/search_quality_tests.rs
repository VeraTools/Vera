//! Search quality integration tests.
//!
//! Verifies symbol lookup accuracy, natural language intent search,
//! cross-file discovery, result structure completeness, and search filters.
//!
//! Uses a realistic multi-language, multi-file test corpus with known
//! symbols, content, and cross-file relationships.

use crate::config::VeraConfig;
use crate::embedding::test_helpers::MockProvider;
use crate::indexing::index_repository;
use crate::parsing;
use crate::retrieval::apply_filters;
use crate::retrieval::bm25::search_bm25_with_stores;
use crate::retrieval::hybrid::fuse_rrf;
use crate::retrieval::vector::search_vector_with_stores;
use crate::storage::bm25::Bm25Index;
use crate::storage::metadata::MetadataStore;
use crate::storage::vector::VectorStore;
use crate::types::{Chunk, Language, SearchFilters, SearchResult, SymbolType};

// ── Test corpus ─────────────────────────────────────────────────────

/// Build a realistic multi-file, multi-language test corpus with known
/// symbols and content for quality testing.
fn build_test_corpus() -> Vec<(String, &'static str, Language)> {
    vec![
        (
            "src/auth.rs".to_string(),
            r#"pub fn authenticate(user: &str, password: &str) -> Result<Token, AuthError> {
    let hash = hash_password(password);
    let stored = get_stored_hash(user)?;
    if constant_time_eq(&hash, &stored) {
        Ok(Token::new(user))
    } else {
        Err(AuthError::InvalidCredentials)
    }
}

pub fn authorize(token: &Token, resource: &str) -> bool {
    token.permissions().contains(&resource.to_string())
}

pub struct Token {
    pub user: String,
    pub permissions: Vec<String>,
    pub expires_at: u64,
}

impl Token {
    pub fn new(user: &str) -> Self {
        Self {
            user: user.to_string(),
            permissions: Vec::new(),
            expires_at: 0,
        }
    }

    pub fn has_permission(&self, resource: &str) -> bool {
        self.permissions.contains(&resource.to_string())
    }
}

pub enum AuthError {
    InvalidCredentials,
    TokenExpired,
    InsufficientPermissions,
}
"#,
            Language::Rust,
        ),
        (
            "src/handler.rs".to_string(),
            r#"use crate::auth::{authenticate, authorize, Token};

pub fn handle_login(request: &Request) -> Response {
    let user = request.body.get("username").unwrap_or_default();
    let pass = request.body.get("password").unwrap_or_default();
    match authenticate(user, pass) {
        Ok(token) => Response::ok(token),
        Err(e) => Response::unauthorized(e),
    }
}

pub fn handle_protected(request: &Request, token: &Token) -> Response {
    if !authorize(token, &request.path) {
        return Response::forbidden();
    }
    process_request(request)
}

pub fn handle_error(err: &AppError) -> Response {
    match err {
        AppError::NotFound(msg) => Response::not_found(msg),
        AppError::BadRequest(msg) => Response::bad_request(msg),
        AppError::Internal(msg) => Response::internal_error(msg),
    }
}

pub struct Request {
    pub path: String,
    pub body: std::collections::HashMap<String, String>,
}

pub struct Response {
    pub status: u16,
    pub body: String,
}
"#,
            Language::Rust,
        ),
        (
            "src/database.py".to_string(),
            r#"class DatabaseConnection:
    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self.connection = None

    def connect(self):
        self.connection = create_connection(self.host, self.port, self.database)
        return self

    def execute_query(self, sql: str, params=None):
        cursor = self.connection.cursor()
        cursor.execute(sql, params or [])
        return cursor.fetchall()

    def close(self):
        if self.connection:
            self.connection.close()

class UserRepository:
    def __init__(self, db: DatabaseConnection):
        self.db = db

    def find_by_username(self, username: str):
        return self.db.execute_query(
            "SELECT * FROM users WHERE username = ?",
            [username]
        )

    def create_user(self, username: str, email: str, password_hash: str):
        return self.db.execute_query(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            [username, email, password_hash]
        )

    def delete_user(self, user_id: int):
        return self.db.execute_query(
            "DELETE FROM users WHERE id = ?",
            [user_id]
        )
"#,
            Language::Python,
        ),
        (
            "src/router.ts".to_string(),
            r#"export interface Route {
    path: string;
    method: string;
    handler: (req: Request) => Response;
}

export class Router {
    private routes: Route[] = [];

    addRoute(path: string, method: string, handler: (req: Request) => Response): void {
        this.routes.push({ path, method, handler });
    }

    match(path: string, method: string): Route | undefined {
        return this.routes.find(r => r.path === path && r.method === method);
    }

    handleRequest(req: Request): Response {
        const route = this.match(req.url, req.method);
        if (!route) {
            return new Response('Not Found', { status: 404 });
        }
        return route.handler(req);
    }
}

export function createRouter(): Router {
    return new Router();
}
"#,
            Language::TypeScript,
        ),
        (
            "src/cache.go".to_string(),
            r#"package cache

type CacheEntry struct {
    Key       string
    Value     interface{}
    ExpiresAt int64
}

type Cache struct {
    entries map[string]*CacheEntry
    maxSize int
}

func NewCache(maxSize int) *Cache {
    return &Cache{
        entries: make(map[string]*CacheEntry),
        maxSize: maxSize,
    }
}

func (c *Cache) Get(key string) (interface{}, bool) {
    entry, ok := c.entries[key]
    if !ok || entry.ExpiresAt < time.Now().Unix() {
        return nil, false
    }
    return entry.Value, true
}

func (c *Cache) Set(key string, value interface{}, ttl int64) {
    if len(c.entries) >= c.maxSize {
        c.evictOldest()
    }
    c.entries[key] = &CacheEntry{
        Key:       key,
        Value:     value,
        ExpiresAt: time.Now().Unix() + ttl,
    }
}

func (c *Cache) Delete(key string) {
    delete(c.entries, key)
}

func (c *Cache) evictOldest() {
    var oldestKey string
    var oldestTime int64 = math.MaxInt64
    for k, v := range c.entries {
        if v.ExpiresAt < oldestTime {
            oldestTime = v.ExpiresAt
            oldestKey = k
        }
    }
    if oldestKey != "" {
        delete(c.entries, oldestKey)
    }
}
"#,
            Language::Go,
        ),
        (
            "src/middleware.rs".to_string(),
            r#"pub fn rate_limiter(max_requests: u32, window_secs: u64) -> impl Fn(&Request) -> bool {
    move |request| {
        let key = request.remote_addr();
        check_rate_limit(key, max_requests, window_secs)
    }
}

pub fn cors_middleware(allowed_origins: &[&str]) -> impl Fn(&mut Response) {
    let origins: Vec<String> = allowed_origins.iter().map(|s| s.to_string()).collect();
    move |response| {
        response.headers.insert(
            "Access-Control-Allow-Origin".to_string(),
            origins.join(", "),
        );
    }
}

pub fn logging_middleware(request: &Request) {
    tracing::info!(
        method = %request.method,
        path = %request.path,
        "incoming request"
    );
}
"#,
            Language::Rust,
        ),
    ]
}

/// Create an in-memory indexed corpus for quality tests.
///
/// Returns (bm25_index, metadata_store, vector_store, chunks, provider,
/// corpus_sources) with all data indexed and ready to search.
/// `corpus_sources` maps file_path → source content for token efficiency tests.
async fn setup_indexed_corpus() -> (
    Bm25Index,
    MetadataStore,
    VectorStore,
    Vec<Chunk>,
    MockProvider,
    std::collections::HashMap<String, String>,
) {
    let dim = 8;
    let provider = MockProvider::new(dim);
    let config = VeraConfig::default();
    let corpus = build_test_corpus();

    let mut all_chunks = Vec::new();
    let mut corpus_sources = std::collections::HashMap::new();
    for (file_path, source, language) in &corpus {
        corpus_sources.insert(file_path.clone(), source.to_string());
        let chunks =
            parsing::parse_and_chunk(source, file_path, *language, &config.indexing).unwrap();
        all_chunks.extend(chunks);
    }

    // Create stores.
    let metadata_store = MetadataStore::open_in_memory().unwrap();
    metadata_store.insert_chunks(&all_chunks).unwrap();

    let vector_store = VectorStore::open_in_memory(dim).unwrap();
    crate::embedding::test_helpers::embed_and_insert_vectors(&vector_store, &provider, &all_chunks)
        .await;

    let bm25_index = Bm25Index::open_in_memory().unwrap();
    bm25_index.insert_chunks(&all_chunks).unwrap();

    (
        bm25_index,
        metadata_store,
        vector_store,
        all_chunks,
        provider,
        corpus_sources,
    )
}

// ── Helper: BM25 search shortcut ────────────────────────────────────

fn bm25_search(
    bm25: &Bm25Index,
    metadata: &MetadataStore,
    query: &str,
    limit: usize,
) -> Vec<SearchResult> {
    search_bm25_with_stores(bm25, metadata, query, limit).unwrap()
}

async fn setup_structural_repo() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().unwrap();
    for (file_path, source, _) in build_test_corpus() {
        let abs = dir.path().join(&file_path);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(abs, source).unwrap();
    }

    let provider = MockProvider::new(8);
    let config = VeraConfig::default();
    index_repository(dir.path(), &provider, &config, "mock-model")
        .await
        .unwrap();
    dir
}

// ── 1. Exact symbol lookup tests (10+ queries) ─────────────────────

#[tokio::test]
async fn symbol_lookup_authenticate_function() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 10);
    assert!(!results.is_empty());
    assert_eq!(
        results[0].symbol_name.as_deref(),
        Some("authenticate"),
        "authenticate should be top-1"
    );
}

#[tokio::test]
async fn symbol_lookup_authorize_function() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authorize", 10);
    assert!(!results.is_empty());
    let top3_names: Vec<_> = results
        .iter()
        .take(3)
        .filter_map(|r| r.symbol_name.as_deref())
        .collect();
    assert!(
        top3_names.contains(&"authorize"),
        "authorize should be in top-3: got {:?}",
        top3_names
    );
}

#[tokio::test]
async fn symbol_lookup_token_struct() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "Token", 10);
    assert!(!results.is_empty());
    let top3_names: Vec<_> = results
        .iter()
        .take(3)
        .filter_map(|r| r.symbol_name.as_deref())
        .collect();
    assert!(
        top3_names.contains(&"Token"),
        "Token struct should be in top-3: got {:?}",
        top3_names
    );
}

#[tokio::test]
async fn symbol_lookup_database_connection_class() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "DatabaseConnection", 10);
    assert!(!results.is_empty());
    // Python class methods are now split into individual chunks, so
    // "DatabaseConnection" appears in the class header gap chunk or
    // within method chunks' content, not as a single Class symbol.
    let top3_have_content = results
        .iter()
        .take(3)
        .any(|r| r.content.contains("DatabaseConnection"));
    assert!(
        top3_have_content,
        "DatabaseConnection should appear in content of top-3 results"
    );
}

#[tokio::test]
async fn symbol_lookup_user_repository() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "UserRepository", 10);
    assert!(!results.is_empty());
    // Python class methods are now split into individual chunks.
    let top3_have_content = results
        .iter()
        .take(3)
        .any(|r| r.content.contains("UserRepository"));
    assert!(
        top3_have_content,
        "UserRepository should appear in content of top-3 results"
    );
}

#[tokio::test]
async fn symbol_lookup_router_class() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "Router", 10);
    assert!(!results.is_empty());
    let top3_names: Vec<_> = results
        .iter()
        .take(3)
        .filter_map(|r| r.symbol_name.as_deref())
        .collect();
    assert!(
        top3_names.contains(&"Router"),
        "Router class should be in top-3: got {:?}",
        top3_names
    );
}

#[tokio::test]
async fn symbol_lookup_cache_struct() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "Cache", 10);
    assert!(!results.is_empty());
    let top3_names: Vec<_> = results
        .iter()
        .take(3)
        .filter_map(|r| r.symbol_name.as_deref())
        .collect();
    assert!(
        top3_names.contains(&"Cache"),
        "Cache struct should be in top-3: got {:?}",
        top3_names
    );
}

#[tokio::test]
async fn symbol_lookup_handle_login() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "handle_login", 10);
    assert!(!results.is_empty());
    assert_eq!(
        results[0].symbol_name.as_deref(),
        Some("handle_login"),
        "handle_login should be top-1"
    );
}

#[tokio::test]
async fn symbol_lookup_rate_limiter() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "rate_limiter", 10);
    assert!(!results.is_empty());
    assert_eq!(
        results[0].symbol_name.as_deref(),
        Some("rate_limiter"),
        "rate_limiter should be top-1"
    );
}

#[tokio::test]
async fn symbol_lookup_find_by_username() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "find_by_username", 10);
    assert!(!results.is_empty());
    // find_by_username is a method inside the UserRepository class chunk,
    // so it may appear as content within the class chunk rather than a
    // separate chunk with that symbol_name.
    let top3_have_content = results
        .iter()
        .take(3)
        .any(|r| r.content.contains("find_by_username"));
    assert!(
        top3_have_content,
        "find_by_username should appear in content of top-3 results"
    );
}

// ── Structural agent-task regressions ─────────────────────────────

#[tokio::test]
async fn references_search_finds_authenticate_handler() {
    let repo = setup_structural_repo().await;
    let results = crate::retrieval::search_callers(
        &crate::indexing::index_dir(repo.path()),
        "authenticate",
        10,
        &SearchFilters::default(),
    )
    .unwrap();

    assert!(
        results
            .iter()
            .any(|result| result.file_path == "src/handler.rs"
                && result.content.contains("authenticate(user, pass)")),
        "expected authenticate callsite in handler.rs, got {results:?}"
    );
}

#[tokio::test]
async fn structural_sql_finds_database_execution_sites() {
    let repo = setup_structural_repo().await;
    let results = crate::retrieval::search_structural(
        &crate::indexing::index_dir(repo.path()),
        crate::retrieval::StructuralSearchKind::SqlQueries,
        None,
        10,
        &SearchFilters::default(),
    )
    .unwrap();

    assert!(
        results
            .iter()
            .any(|result| result.file_path == "src/database.py"),
        "expected SQL execution site in database.py, got {results:?}"
    );
}

#[tokio::test]
async fn symbol_lookup_handle_error() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "handle_error", 10);
    assert!(!results.is_empty());
    assert_eq!(
        results[0].symbol_name.as_deref(),
        Some("handle_error"),
        "handle_error should be top-1"
    );
}

#[tokio::test]
async fn symbol_lookup_create_router() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "createRouter", 10);
    assert!(!results.is_empty());
    assert_eq!(
        results[0].symbol_name.as_deref(),
        Some("createRouter"),
        "createRouter should be top-1"
    );
}

/// Aggregate: verify 80%+ top-1 accuracy across 12 symbol lookups.
#[tokio::test]
async fn symbol_lookup_top1_accuracy_80_percent() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;

    let queries_with_expected = [
        ("authenticate", "authenticate"),
        ("authorize", "authorize"),
        ("execute_query", "execute_query"),
        ("find_by_username", "find_by_username"),
        ("handle_login", "handle_login"),
        ("rate_limiter", "rate_limiter"),
        ("handle_error", "handle_error"),
        ("createRouter", "createRouter"),
        ("cors_middleware", "cors_middleware"),
        ("handle_protected", "handle_protected"),
        ("logging_middleware", "logging_middleware"),
        ("NewCache", "NewCache"),
    ];

    let mut top1_hits = 0;
    let mut top3_hits = 0;
    for (query, expected_name) in &queries_with_expected {
        let results = bm25_search(&bm25, &meta, query, 10);
        if !results.is_empty() && results[0].symbol_name.as_deref() == Some(expected_name) {
            top1_hits += 1;
        }
        let top3_names: Vec<_> = results
            .iter()
            .take(3)
            .filter_map(|r| r.symbol_name.as_deref())
            .collect();
        if top3_names.contains(expected_name) {
            top3_hits += 1;
        }
    }

    let total = queries_with_expected.len();
    let top1_accuracy = top1_hits as f64 / total as f64;
    let top3_accuracy = top3_hits as f64 / total as f64;

    assert!(
        top1_accuracy >= 0.80,
        "top-1 accuracy {:.0}% ({top1_hits}/{total}) should be >= 80%",
        top1_accuracy * 100.0
    );
    assert!(
        top3_accuracy >= 0.90,
        "top-3 accuracy {:.0}% ({top3_hits}/{total}) should be >= 90%",
        top3_accuracy * 100.0
    );
}

// ── 2. Natural language intent search (5+ queries) ──────────────────

#[tokio::test]
async fn intent_search_authentication_logic() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(
        &bm25,
        &meta,
        "authentication logic password verification",
        5,
    );
    assert!(!results.is_empty());
    let has_auth = results
        .iter()
        .any(|r| r.file_path.contains("auth") || r.content.contains("authenticate"));
    assert!(
        has_auth,
        "authentication intent should find auth-related code"
    );
}

#[tokio::test]
async fn intent_search_error_handling() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "error handling response", 5);
    assert!(!results.is_empty());
    let has_error = results
        .iter()
        .any(|r| r.content.contains("error") || r.content.contains("Error"));
    assert!(has_error, "error handling intent should find error code");
}

#[tokio::test]
async fn intent_search_database_queries() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "database query SQL", 5);
    assert!(!results.is_empty());
    let has_db = results
        .iter()
        .any(|r| r.file_path.contains("database") || r.content.contains("execute_query"));
    assert!(has_db, "database query intent should find DB code");
}

#[tokio::test]
async fn intent_search_http_routing() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "request routing handler", 5);
    assert!(!results.is_empty());
    let has_routing = results.iter().any(|r| {
        r.content.contains("route")
            || r.content.contains("request")
            || r.content.contains("handler")
    });
    assert!(has_routing, "routing intent should find routing code");
}

#[tokio::test]
async fn intent_search_caching() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "caching entries eviction", 5);
    assert!(!results.is_empty());
    let has_cache = results
        .iter()
        .any(|r| r.file_path.contains("cache") || r.content.contains("Cache"));
    assert!(has_cache, "caching intent should find cache code");
}

#[tokio::test]
async fn intent_search_user_management() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "user management create delete", 5);
    assert!(!results.is_empty());
    let has_user = results.iter().any(|r| {
        r.content.contains("user")
            || r.content.contains("username")
            || r.content.contains("create_user")
    });
    assert!(has_user, "user management intent should find user code");
}

// ── 3. Cross-file discovery ─────────────────────────────────────────

#[tokio::test]
async fn cross_file_auth_concept() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    // "authenticate" concept spans auth.rs and handler.rs
    let results = bm25_search(&bm25, &meta, "authenticate", 10);
    let unique_files: std::collections::HashSet<_> =
        results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
        unique_files.len() >= 2,
        "authenticate should return results from 2+ files, got: {:?}",
        unique_files
    );
}

#[tokio::test]
async fn cross_file_request_handling() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    // "request" concept spans handler.rs, router.ts, middleware.rs
    let results = bm25_search(&bm25, &meta, "request", 10);
    let unique_files: std::collections::HashSet<_> =
        results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
        unique_files.len() >= 2,
        "request should return results from 2+ files, got: {:?}",
        unique_files
    );
}

#[tokio::test]
async fn cross_file_user_concept() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    // "user" concept spans auth.rs, database.py, handler.rs
    let results = bm25_search(&bm25, &meta, "user", 10);
    let unique_files: std::collections::HashSet<_> =
        results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
        unique_files.len() >= 2,
        "user should return results from 2+ files, got: {:?}",
        unique_files
    );
}

// ── 4. Result structure completeness ────────────────────────────────

#[tokio::test]
async fn result_structure_all_fields_present() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 10);
    assert!(!results.is_empty());

    for result in &results {
        // Required fields always present.
        assert!(
            !result.file_path.is_empty(),
            "file_path should be non-empty"
        );
        assert!(
            result.line_start > 0,
            "line_start should be 1-based positive"
        );
        assert!(
            result.line_end >= result.line_start,
            "line_end >= line_start"
        );
        assert!(!result.content.is_empty(), "content should be non-empty");
        assert!(result.score > 0.0, "score should be positive");
        // language should be a valid variant (it's always set).
    }
}

#[tokio::test]
async fn result_structure_symbol_fields_for_named_symbols() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 10);
    assert!(!results.is_empty());

    // The top result for "authenticate" should have symbol metadata.
    let top = &results[0];
    assert!(
        top.symbol_name.is_some(),
        "symbol_name should be present for named symbol"
    );
    assert!(
        top.symbol_type.is_some(),
        "symbol_type should be present for named symbol"
    );
    assert_eq!(top.symbol_name.as_deref(), Some("authenticate"));
    assert_eq!(top.symbol_type, Some(SymbolType::Function));
}

#[tokio::test]
async fn result_scores_descending() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "request handler", 10);

    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "scores must be descending at position {i}: {} >= {}",
            results[i - 1].score,
            results[i].score
        );
    }
}

#[tokio::test]
async fn result_json_serialization_complete() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 5);
    assert!(!results.is_empty());

    let json = serde_json::to_string_pretty(&results).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let arr = parsed.as_array().unwrap();
    assert!(!arr.is_empty());

    // Context capsule schema: every result must have all 9 fields (including part_index).
    let expected_keys: std::collections::BTreeSet<&str> = [
        "file_path",
        "line_start",
        "line_end",
        "content",
        "language",
        "score",
        "symbol_name",
        "symbol_type",
        "part_index",
    ]
    .into_iter()
    .collect();

    for item in arr {
        let obj = item.as_object().unwrap();
        let keys: std::collections::BTreeSet<&str> = obj.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys, expected_keys,
            "every result must have exactly the context capsule fields"
        );

        // Core fields are always their expected types.
        assert!(item["file_path"].is_string());
        assert!(item["line_start"].is_number());
        assert!(item["line_end"].is_number());
        assert!(item["content"].is_string());
        assert!(item["language"].is_string());
        assert!(item["score"].is_number());

        // symbol_name and symbol_type: either string or null, never missing.
        assert!(
            item["symbol_name"].is_string() || item["symbol_name"].is_null(),
            "symbol_name must be string or null, got: {:?}",
            item["symbol_name"]
        );
        assert!(
            item["symbol_type"].is_string() || item["symbol_type"].is_null(),
            "symbol_type must be string or null, got: {:?}",
            item["symbol_type"]
        );
    }

    // The top result (authenticate) should have symbol fields.
    let top = &arr[0];
    assert!(top["symbol_name"].is_string());
    assert!(top["symbol_type"].is_string());
}

#[tokio::test]
async fn result_snippet_fidelity() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 5);
    assert!(!results.is_empty());

    let top = &results[0];
    // The content of the authenticate function should contain key identifiers.
    assert!(
        top.content.contains("authenticate"),
        "snippet should contain the function name"
    );
    assert!(
        top.content.contains("password"),
        "snippet should contain parameter names"
    );
}

// ── 5. Search filter tests ──────────────────────────────────────────

#[tokio::test]
async fn filter_by_language_rust_only() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "request", 20);

    let filters = SearchFilters {
        language: Some("rust".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    assert!(!filtered.is_empty());
    for result in &filtered {
        assert_eq!(
            result.language,
            Language::Rust,
            "all filtered results should be Rust"
        );
    }
}

#[tokio::test]
async fn filter_by_language_python_only() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "user", 20);

    let filters = SearchFilters {
        language: Some("python".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    assert!(!filtered.is_empty());
    for result in &filtered {
        assert_eq!(
            result.language,
            Language::Python,
            "all filtered results should be Python"
        );
    }
}

#[tokio::test]
async fn filter_by_path_glob() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "request", 20);

    let filters = SearchFilters {
        path_glob: vec!["**/*.rs".to_string()],
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    assert!(!filtered.is_empty());
    for result in &filtered {
        assert!(
            result.file_path.ends_with(".rs"),
            "all filtered results should be .rs files, got: {}",
            result.file_path
        );
    }
}

#[tokio::test]
async fn filter_by_path_glob_specific_directory() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "function class struct", 30);

    let filters = SearchFilters {
        path_glob: vec!["src/auth*".to_string()],
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 30);

    for result in &filtered {
        assert!(
            result.file_path.starts_with("src/auth"),
            "all filtered results should match src/auth*, got: {}",
            result.file_path
        );
    }
}

#[tokio::test]
async fn filter_by_symbol_type_function() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "handle request", 20);

    let filters = SearchFilters {
        symbol_type: Some("function".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    for result in &filtered {
        assert!(
            matches!(
                result.symbol_type,
                Some(SymbolType::Function) | Some(SymbolType::Method)
            ),
            "all filtered results should be functions or methods, got: {:?}",
            result.symbol_type
        );
    }
}

#[tokio::test]
async fn filter_by_symbol_type_class() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    // Search for content that exists in TypeScript class (Router).
    let results = bm25_search(&bm25, &meta, "Router routes handler", 20);

    let filters = SearchFilters {
        symbol_type: Some("class".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    assert!(!filtered.is_empty(), "should find class-type results");
    for result in &filtered {
        assert_eq!(
            result.symbol_type,
            Some(SymbolType::Class),
            "all filtered results should be classes"
        );
    }
}

#[tokio::test]
async fn filter_limit_caps_results() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "function", 20);

    let filters = SearchFilters::default();
    let limited = apply_filters(results, &filters, 3);

    assert!(
        limited.len() <= 3,
        "limit should cap results to 3, got {}",
        limited.len()
    );
}

#[tokio::test]
async fn filter_combined_lang_and_type() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate handler request", 20);

    let filters = SearchFilters {
        language: Some("rust".to_string()),
        symbol_type: Some("function".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    for result in &filtered {
        assert_eq!(result.language, Language::Rust);
        assert_eq!(result.symbol_type, Some(SymbolType::Function));
    }
}

#[tokio::test]
async fn filter_combined_path_and_lang() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "user", 20);

    let filters = SearchFilters {
        language: Some("python".to_string()),
        path_glob: vec!["**/*.py".to_string()],
        ..Default::default()
    };
    let filtered = apply_filters(results, &filters, 20);

    for result in &filtered {
        assert_eq!(result.language, Language::Python);
        assert!(result.file_path.ends_with(".py"));
    }
}

// ── 6. Hybrid RRF + filter integration ──────────────────────────────

#[tokio::test]
async fn hybrid_rrf_with_filters() {
    let (bm25, meta, vec_store, _, provider, _) = setup_indexed_corpus().await;

    // Get BM25 results.
    let bm25_results = search_bm25_with_stores(&bm25, &meta, "authenticate", 20).unwrap();

    // Get vector results.
    let vec_results = search_vector_with_stores(&vec_store, &meta, &provider, "authenticate", 20)
        .await
        .unwrap();

    // Fuse via RRF.
    let fused = fuse_rrf(&bm25_results, &vec_results, 60.0, 20);

    // Apply language filter.
    let filters = SearchFilters {
        language: Some("rust".to_string()),
        ..Default::default()
    };
    let filtered = apply_filters(fused, &filters, 10);

    assert!(!filtered.is_empty());
    for result in &filtered {
        assert_eq!(result.language, Language::Rust);
    }

    // Scores should still be descending after filtering.
    for i in 1..filtered.len() {
        assert!(
            filtered[i - 1].score >= filtered[i].score,
            "scores should be descending after filtering"
        );
    }
}

// ── 7. Context capsule schema consistency ───────────────────────────

#[tokio::test]
async fn context_capsule_schema_consistency_across_queries() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;

    // Run multiple diverse queries to get a mix of results with/without symbols.
    let queries = [
        "authenticate",
        "request handler",
        "database query",
        "caching",
        "user",
    ];

    let expected_keys: std::collections::BTreeSet<&str> = [
        "file_path",
        "line_start",
        "line_end",
        "content",
        "language",
        "score",
        "symbol_name",
        "symbol_type",
        "part_index",
    ]
    .into_iter()
    .collect();

    let mut total_results = 0;

    for query in &queries {
        let results = bm25_search(&bm25, &meta, query, 10);
        let json = serde_json::to_string_pretty(&results).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().unwrap();

        for item in arr {
            total_results += 1;
            let obj = item.as_object().unwrap();
            let keys: std::collections::BTreeSet<&str> = obj.keys().map(|k| k.as_str()).collect();
            assert_eq!(
                keys, expected_keys,
                "query '{query}': all results must have the same schema"
            );

            // Type consistency: core fields.
            assert!(item["file_path"].is_string());
            assert!(item["line_start"].is_u64());
            assert!(item["line_end"].is_u64());
            assert!(item["content"].is_string());
            assert!(item["language"].is_string());
            assert!(item["score"].is_f64());

            // Nullable fields: string or null.
            assert!(
                item["symbol_name"].is_string() || item["symbol_name"].is_null(),
                "symbol_name type inconsistency"
            );
            assert!(
                item["symbol_type"].is_string() || item["symbol_type"].is_null(),
                "symbol_type type inconsistency"
            );
        }
    }

    assert!(
        total_results >= 10,
        "should have validated 10+ results, got {total_results}"
    );
}

#[tokio::test]
async fn context_capsule_content_not_truncated() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 5);
    assert!(!results.is_empty());

    let top = &results[0];
    // The authenticate function has a body with multiple lines.
    // Content should contain the complete symbol body.
    assert!(
        top.content.contains("authenticate"),
        "content should contain function name"
    );
    assert!(
        top.content.contains("password"),
        "content should contain parameter"
    );
    // Verify the content spans the declared line range.
    let content_lines = top.content.lines().count();
    let declared_lines = (top.line_end - top.line_start + 1) as usize;
    assert_eq!(
        content_lines, declared_lines,
        "content line count ({content_lines}) should match declared range ({declared_lines})"
    );
}

#[tokio::test]
async fn context_capsule_token_efficiency() {
    let (bm25, meta, _, _, _, corpus_sources) = setup_indexed_corpus().await;

    // Token efficiency: the content fields of search results should be
    // significantly smaller than the full source files they reference.
    // We compare content-only character count vs full file character count.
    // In a real scenario with large repos, this ratio is typically <30%.
    // With our small test corpus we verify the principle: content < files.
    let queries = [
        "authenticate",
        "request handler",
        "database connection",
        "caching entries",
        "user management",
    ];

    let mut total_content_chars = 0usize;
    let mut total_file_chars = 0usize;

    for query in &queries {
        let results = bm25_search(&bm25, &meta, query, 5);
        if results.is_empty() {
            continue;
        }

        // Sum the content field sizes (the actual code payload).
        for result in &results {
            total_content_chars += result.content.len();
        }

        // Collect unique files referenced by results and sum their sizes.
        let unique_files: std::collections::HashSet<_> =
            results.iter().map(|r| r.file_path.as_str()).collect();
        for file_path in &unique_files {
            if let Some(source) = corpus_sources.get(*file_path) {
                total_file_chars += source.len();
            }
        }
    }

    assert!(total_content_chars > 0, "should have some results");
    assert!(total_file_chars > 0, "should have matching files");

    let ratio = total_content_chars as f64 / total_file_chars as f64;
    // Content extracted from search results must be smaller than reading
    // entire files. The full ≤30% target applies to real repos; for our
    // small test corpus, content should be strictly less than full files.
    assert!(
        ratio < 1.0,
        "search content should be more compact than full files, got ratio {:.2} \
         (content={total_content_chars}, files={total_file_chars})",
        ratio
    );
}

#[tokio::test]
async fn context_capsule_json_parseable() {
    let (bm25, meta, _, _, _, _) = setup_indexed_corpus().await;
    let results = bm25_search(&bm25, &meta, "authenticate", 10);
    assert!(!results.is_empty());

    // Verify JSON is parseable.
    let json = serde_json::to_string_pretty(&results).unwrap();

    // Parse as generic JSON value (simulates jq).
    let _: serde_json::Value =
        serde_json::from_str(&json).expect("JSON output should parse as a valid JSON value");

    // Parse back into typed SearchResult vec.
    let round_trip: Vec<SearchResult> =
        serde_json::from_str(&json).expect("JSON output should round-trip back to SearchResult");
    assert_eq!(round_trip.len(), results.len());
}
