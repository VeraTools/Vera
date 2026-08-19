//! Tantivy-based BM25 full-text index over chunk content.
//!
//! Provides keyword search with BM25 scoring. Indexes chunk content,
//! symbol names, and file paths for comprehensive text search.

use std::path::Path;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument, doc};

use crate::chunk_text;

/// Tantivy-backed BM25 full-text search index.
pub struct Bm25Index {
    index: Index,
    schema: Bm25Schema,
}

/// Pre-resolved schema field handles for efficient access.
#[derive(Clone)]
struct Bm25Schema {
    schema: Schema,
    chunk_id: tantivy::schema::Field,
    file_path: tantivy::schema::Field,
    filename: tantivy::schema::Field,
    path_tokens: tantivy::schema::Field,
    content: tantivy::schema::Field,
    symbol_name: tantivy::schema::Field,
    language: tantivy::schema::Field,
}

/// A single BM25 search result.
#[derive(Debug, Clone)]
pub struct Bm25SearchResult {
    /// The chunk ID.
    pub chunk_id: String,
    /// BM25 relevance score (higher is better).
    pub score: f32,
}

/// Writer heap size for Tantivy (50MB).
const WRITER_HEAP_SIZE: usize = 50_000_000;

/// Custom tokenizer name: SimpleTokenizer + LowerCaser + English stemmer.
///
/// English stemming lets "dependency" match "dependencies", "configure" match
/// "configuration", etc. This is standard search-engine behaviour and fixes
/// a large class of recall failures for natural-language queries.
const STEMMED_TOKENIZER: &str = "code_en_stem";

fn register_stemmed_tokenizer(index: &Index) {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(tantivy::tokenizer::Stemmer::new(
            tantivy::tokenizer::Language::English,
        ))
        .build();
    index.tokenizers().register(STEMMED_TOKENIZER, analyzer);
}

impl Bm25Index {
    /// Open (or create) a BM25 index at the given directory path.
    pub fn open(index_dir: &Path) -> Result<Self> {
        let schema = build_schema();

        let index = if index_dir.exists() && index_dir.join("meta.json").exists() {
            Index::open_in_dir(index_dir)
                .with_context(|| format!("failed to open BM25 index: {}", index_dir.display()))?
        } else {
            std::fs::create_dir_all(index_dir)
                .with_context(|| format!("failed to create BM25 dir: {}", index_dir.display()))?;
            Index::create_in_dir(index_dir, schema.schema.clone())
                .with_context(|| format!("failed to create BM25 index: {}", index_dir.display()))?
        };

        register_stemmed_tokenizer(&index);
        Ok(Self { index, schema })
    }

    /// Create an in-memory BM25 index (useful for testing).
    pub fn open_in_memory() -> Result<Self> {
        let schema = build_schema();
        let index = Index::create_in_ram(schema.schema.clone());
        register_stemmed_tokenizer(&index);
        Ok(Self { index, schema })
    }

    /// Insert a batch of documents into the index.
    ///
    /// Each document is a tuple of (chunk_id, file_path, content, symbol_name, language).
    pub fn insert_batch(&self, docs: &[Bm25Document<'_>]) -> Result<()> {
        let mut writer: IndexWriter = self
            .index
            .writer(WRITER_HEAP_SIZE)
            .context("failed to create BM25 index writer")?;

        for doc_data in docs {
            let sym_name = doc_data.symbol_name.unwrap_or("");
            let chunk = crate::types::Chunk {
                id: doc_data.chunk_id.to_string(),
                file_path: doc_data.file_path.to_string(),
                line_start: 0,
                line_end: 0,
                content: doc_data.content.to_string(),
                language: doc_data
                    .language
                    .parse()
                    .unwrap_or(crate::types::Language::Unknown),
                symbol_type: None,
                symbol_name: doc_data.symbol_name.map(ToString::to_string),
            };
            let searchable_text = chunk_text::build_bm25_text(&chunk);
            writer
                .add_document(doc!(
                    self.schema.chunk_id => doc_data.chunk_id.to_string(),
                    self.schema.file_path => doc_data.file_path.to_string(),
                    self.schema.filename => chunk_text::file_name(doc_data.file_path).to_string(),
                    self.schema.path_tokens => chunk_text::normalize_path_tokens(doc_data.file_path),
                    self.schema.content => searchable_text,
                    self.schema.symbol_name => sym_name.to_string(),
                    self.schema.language => doc_data.language.to_string(),
                ))
                .context("failed to add document to BM25 index")?;
        }

        writer.commit().context("failed to commit BM25 index")?;
        // Wait for merging to complete.
        writer
            .wait_merging_threads()
            .map_err(|e| anyhow::anyhow!("BM25 merge failed: {e}"))?;

        Ok(())
    }

    /// Insert all chunks as BM25 documents in one batch.
    pub fn insert_chunks(&self, chunks: &[crate::types::Chunk]) -> Result<()> {
        let lang_strings: Vec<String> = chunks.iter().map(|c| c.language.to_string()).collect();
        let docs: Vec<Bm25Document<'_>> = chunks
            .iter()
            .zip(lang_strings.iter())
            .map(|(c, lang)| Bm25Document {
                chunk_id: &c.id,
                file_path: &c.file_path,
                content: &c.content,
                symbol_name: c.symbol_name.as_deref(),
                language: lang,
            })
            .collect();
        self.insert_batch(&docs)
    }

    /// Search the index with a text query.
    ///
    /// Searches across content and symbol_name fields.
    pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<Bm25SearchResult>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("failed to create BM25 reader")?;

        let searcher = reader.searcher();
        let mut query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.schema.filename,
                self.schema.path_tokens,
                self.schema.symbol_name,
                self.schema.content,
            ],
        );
        let path_weighted = looks_path_weighted(query_text);
        query_parser.set_field_boost(self.schema.content, 1.0);
        query_parser.set_field_boost(
            self.schema.symbol_name,
            if path_weighted { 2.0 } else { 3.0 },
        );
        query_parser.set_field_boost(
            self.schema.path_tokens,
            if path_weighted { 5.0 } else { 1.5 },
        );
        query_parser.set_field_boost(self.schema.filename, if path_weighted { 8.0 } else { 2.0 });

        // Sanitize query for tantivy: strip characters that the query parser
        // interprets as operators (e.g. `:` as field separator, `(`, `)`, etc.).
        let sanitized = sanitize_query(query_text);
        let query = query_parser
            .parse_query(&sanitized)
            .with_context(|| format!("failed to parse BM25 query: {query_text}"))?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .context("BM25 search failed")?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_addr) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_addr)
                .context("failed to retrieve BM25 doc")?;
            let chunk_id = doc
                .get_first(self.schema.chunk_id)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            results.push(Bm25SearchResult { chunk_id, score });
        }

        Ok(results)
    }

    /// Count total documents in the index.
    pub fn doc_count(&self) -> Result<u64> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("failed to create BM25 reader for count")?;
        let searcher = reader.searcher();
        Ok(searcher.num_docs())
    }

    /// Delete all documents matching a chunk ID.
    pub fn delete_by_chunk_id(&self, chunk_id: &str) -> Result<()> {
        let mut writer: IndexWriter = self
            .index
            .writer(WRITER_HEAP_SIZE)
            .context("failed to create BM25 writer for delete")?;

        let term = tantivy::Term::from_field_text(self.schema.chunk_id, chunk_id);
        writer.delete_term(term);
        writer.commit().context("failed to commit BM25 delete")?;
        writer
            .wait_merging_threads()
            .map_err(|e| anyhow::anyhow!("BM25 merge after delete failed: {e}"))?;

        Ok(())
    }

    /// Delete all documents for a given file path.
    pub fn delete_by_file(&self, file_path: &str) -> Result<()> {
        self.delete_by_files(std::slice::from_ref(&file_path))
    }

    /// Delete all documents for every given file path, using one writer.
    ///
    /// The per-call cost here is the writer lifecycle, not the deletion:
    /// allocating a `WRITER_HEAP_SIZE` writer, committing a segment and
    /// joining the merge threads costs on the order of tens of milliseconds
    /// regardless of how many terms are deleted. Callers removing many files
    /// must therefore batch, or they pay that fixed cost per file.
    pub fn delete_by_files(&self, file_paths: &[&str]) -> Result<()> {
        if file_paths.is_empty() {
            return Ok(());
        }

        let mut writer: IndexWriter = self
            .index
            .writer(WRITER_HEAP_SIZE)
            .context("failed to create BM25 writer for file delete")?;

        for file_path in file_paths {
            let term = tantivy::Term::from_field_text(self.schema.file_path, file_path);
            writer.delete_term(term);
        }
        writer
            .commit()
            .context("failed to commit BM25 file delete")?;
        writer
            .wait_merging_threads()
            .map_err(|e| anyhow::anyhow!("BM25 merge after file delete failed: {e}"))?;

        Ok(())
    }

    /// Clear the entire index (drop and recreate).
    pub fn clear(&self) -> Result<()> {
        let mut writer: IndexWriter = self
            .index
            .writer(WRITER_HEAP_SIZE)
            .context("failed to create BM25 writer for clear")?;
        writer
            .delete_all_documents()
            .context("failed to clear BM25 index")?;
        writer.commit().context("failed to commit BM25 clear")?;
        writer
            .wait_merging_threads()
            .map_err(|e| anyhow::anyhow!("BM25 merge after clear failed: {e}"))?;
        Ok(())
    }
}

/// Data for a single BM25 document insertion.
pub struct Bm25Document<'a> {
    pub chunk_id: &'a str,
    pub file_path: &'a str,
    pub content: &'a str,
    pub symbol_name: Option<&'a str>,
    pub language: &'a str,
}

/// Build the Tantivy schema for BM25 indexing.
fn build_schema() -> Bm25Schema {
    let mut builder = Schema::builder();

    // Stemmed text: indexed with English stemmer so "dependency" matches "dependencies".
    let stemmed_indexing = TextFieldIndexing::default()
        .set_tokenizer(STEMMED_TOKENIZER)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let stemmed = TextOptions::default().set_indexing_options(stemmed_indexing.clone());
    let stemmed_stored = stemmed.clone() | STORED;

    // chunk_id and file_path use STRING (exact match, no tokenization) for deletion.
    let chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
    let file_path = builder.add_text_field("file_path", STRING | STORED);
    let filename = builder.add_text_field("filename", stemmed_stored.clone());
    let path_tokens = builder.add_text_field("path_tokens", stemmed.clone());
    let content = builder.add_text_field("content", stemmed_stored.clone());
    let symbol_name = builder.add_text_field("symbol_name", stemmed_stored);
    let language = builder.add_text_field("language", stemmed);
    let schema = builder.build();

    Bm25Schema {
        schema,
        chunk_id,
        file_path,
        filename,
        path_tokens,
        content,
        symbol_name,
        language,
    }
}

fn sanitize_query(query_text: &str) -> String {
    query_text
        .chars()
        .map(|c| match c {
            ':' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '~' | '!' | '\'' | '"' | '`' => ' ',
            _ => c,
        })
        .collect()
}

fn looks_path_weighted(query_text: &str) -> bool {
    let lower = query_text.trim().to_ascii_lowercase();
    lower.contains('/')
        || lower.contains('\\')
        || lower.contains(".toml")
        || lower.contains(".json")
        || lower.contains(".yaml")
        || lower.contains(".yml")
        || lower.contains(".ini")
        || lower.contains(".conf")
        || lower.contains("dockerfile")
        || lower.contains("makefile")
        || lower.contains("cmakelists.txt")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<Bm25Document<'static>> {
        vec![
            Bm25Document {
                chunk_id: "src/main.rs:0",
                file_path: "src/main.rs",
                content: "fn main() {\n    println!(\"Hello, world!\");\n}",
                symbol_name: Some("main"),
                language: "rust",
            },
            Bm25Document {
                chunk_id: "src/main.rs:1",
                file_path: "src/main.rs",
                content: "struct Config {\n    name: String,\n    port: u16,\n}",
                symbol_name: Some("Config"),
                language: "rust",
            },
            Bm25Document {
                chunk_id: "src/lib.py:0",
                file_path: "src/lib.py",
                content: "def hello_world():\n    print('Hello from Python')",
                symbol_name: Some("hello_world"),
                language: "python",
            },
            Bm25Document {
                chunk_id: "src/server.ts:0",
                file_path: "src/server.ts",
                content: "function handleRequest(req: Request): Response {\n    return new Response('ok');\n}",
                symbol_name: Some("handleRequest"),
                language: "typescript",
            },
            Bm25Document {
                chunk_id: "Cargo.toml:0",
                file_path: "Cargo.toml",
                content: "[workspace]\nmembers = [\"crates/vera-core\"]\nresolver = \"2\"",
                symbol_name: Some("Cargo.toml"),
                language: "toml",
            },
            Bm25Document {
                chunk_id: "fuzz/Cargo.toml:0",
                file_path: "fuzz/Cargo.toml",
                content: "[package]\nname = \"vera-fuzz\"\nversion = \"0.1.0\"",
                symbol_name: Some("Cargo.toml"),
                language: "toml",
            },
            Bm25Document {
                chunk_id: "turbo.json:0",
                file_path: "turbo.json",
                content: "{ \"pipeline\": { \"build\": { \"dependsOn\": [\"^build\"] } } }",
                symbol_name: Some("turbo.json"),
                language: "json",
            },
            Bm25Document {
                chunk_id: "crates/turbo/src/pipeline.ts:0",
                file_path: "crates/turbo/src/pipeline.ts",
                content: "export type Pipeline = { dependsOn?: string[]; outputs?: string[] }",
                symbol_name: Some("Pipeline"),
                language: "typescript",
            },
        ]
    }

    #[test]
    fn insert_and_count() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();
        assert_eq!(index.doc_count().unwrap(), 8);
    }

    #[test]
    fn keyword_search_finds_content() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("println hello world", 10).unwrap();
        assert!(
            !results.is_empty(),
            "should find documents matching keywords"
        );

        // The main function chunk contains "println" and "Hello, world!"
        let main_found = results.iter().any(|r| r.chunk_id == "src/main.rs:0");
        assert!(main_found, "should find the main function chunk");
    }

    #[test]
    fn symbol_name_search() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("Config", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].chunk_id, "src/main.rs:1",
            "Config struct should be top result for 'Config' query"
        );
    }

    #[test]
    fn search_returns_ranked_scores() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("hello", 10).unwrap();
        assert!(results.len() >= 2, "should find multiple hello matches");

        // Scores should be in descending order.
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores should be descending"
            );
        }
    }

    #[test]
    fn search_respects_limit() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("hello", 1).unwrap();
        assert!(results.len() <= 1);
    }

    #[test]
    fn search_no_results() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("xyznonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn delete_by_files_removes_every_path_and_keeps_the_rest() {
        let index = Bm25Index::open_in_memory().unwrap();
        let mut docs = sample_docs();
        docs.push(Bm25Document {
            chunk_id: "src/keep.rs:0",
            file_path: "src/keep.rs",
            content: "fn hello_survivor() {}",
            symbol_name: Some("hello_survivor"),
            language: "rust",
        });
        index.insert_batch(&docs).unwrap();
        let before = index.doc_count().unwrap();

        // src/main.rs has two documents; deleting it and one other path must
        // take all of theirs and none of anyone else's.
        index
            .delete_by_files(&["src/main.rs", "src/lib.py"])
            .unwrap();

        assert_eq!(index.doc_count().unwrap(), before - 3);

        // "hello" occurs in both deleted paths and in src/keep.rs, so every
        // path assertion below can fail while the result set stays non-empty.
        let hits = index.search("hello", 50).unwrap();
        assert!(
            hits.iter()
                .any(|hit| hit.chunk_id.starts_with("src/keep.rs:")),
            "the unrelated hello document must survive"
        );
        for path in ["src/main.rs:", "src/lib.py:"] {
            assert!(
                hits.iter().all(|hit| !hit.chunk_id.starts_with(path)),
                "{path} should have no documents left, got {:?}",
                hits.iter().map(|h| &h.chunk_id).collect::<Vec<_>>()
            );
        }
        // An unrelated path is untouched.
        let remaining = index.search("server", 50).unwrap();
        assert!(
            remaining
                .iter()
                .any(|hit| hit.chunk_id.starts_with("src/server.ts:")),
            "unrelated documents must survive"
        );
    }

    #[test]
    fn delete_by_files_with_no_paths_is_a_no_op() {
        // The update path calls this unconditionally, so an update with
        // nothing deleted or modified must not touch the index — and must not
        // pay for a writer to do nothing.
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();
        let before = index.doc_count().unwrap();

        index.delete_by_files(&[]).unwrap();

        assert_eq!(index.doc_count().unwrap(), before);
    }

    #[test]
    fn delete_by_chunk_id() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();
        assert_eq!(index.doc_count().unwrap(), 8);

        index.delete_by_chunk_id("src/main.rs:0").unwrap();
        assert_eq!(index.doc_count().unwrap(), 7);
    }

    #[test]
    fn clear_index() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        index.clear().unwrap();
        assert_eq!(index.doc_count().unwrap(), 0);
    }

    #[test]
    fn handlerequest_found_by_request() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index.search("handleRequest", 10).unwrap();
        assert!(!results.is_empty());
        let found = results.iter().any(|r| r.chunk_id == "src/server.ts:0");
        assert!(found, "handleRequest should be found");
    }

    #[test]
    fn filename_queries_rank_config_files() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index
            .search("Cargo.toml workspace configuration", 10)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].chunk_id, "Cargo.toml:0");
    }

    #[test]
    fn path_boost_prefers_root_config_over_nested_match() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index
            .search("Cargo.toml workspace configuration", 10)
            .unwrap();
        assert_eq!(results[0].chunk_id, "Cargo.toml:0");
    }

    #[test]
    fn path_boost_prefers_turbo_json_over_pipeline_symbol() {
        let index = Bm25Index::open_in_memory().unwrap();
        index.insert_batch(&sample_docs()).unwrap();

        let results = index
            .search("turbo.json pipeline configuration", 10)
            .unwrap();
        assert_eq!(results[0].chunk_id, "turbo.json:0");
    }

    #[test]
    fn stemming_matches_morphological_variants() {
        let index = Bm25Index::open_in_memory().unwrap();
        index
            .insert_batch(&[Bm25Document {
                chunk_id: "deps.py:0",
                file_path: "fastapi/dependencies/utils.py",
                content: "def solve_dependencies(dependant, sub_dependant):\n    return resolved",
                symbol_name: Some("solve_dependencies"),
                language: "python",
            }])
            .unwrap();

        // "dependency" should match "dependencies" via English stemming.
        let results = index.search("dependency injection", 10).unwrap();
        assert!(
            !results.is_empty(),
            "stemming should match 'dependency' to 'dependencies'"
        );
    }

    #[test]
    fn sanitize_query_strips_possessive_apostrophes() {
        assert_eq!(
            sanitize_query("how pandoc's app module orchestrates"),
            "how pandoc s app module orchestrates"
        );
    }
}
