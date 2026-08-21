//! `vera search <query>` — Search the indexed codebase.

use anyhow::bail;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};
use vera_core::config::{InferenceBackend, VeraConfig};
use vera_core::retrieval::search_service::{SearchContext, SearchTimings};
use vera_core::types::{SearchFilters, SearchResult};

use crate::helpers::{load_runtime_config, output_results, prepare_indexed_search};

/// Run the `vera search <query>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    queries: &[String],
    intent: Option<&str>,
    limit: Option<usize>,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
    timing: bool,
    deep: bool,
    git_scope: Option<vera_core::git_scope::GitScope>,
    compact: bool,
    backend: InferenceBackend,
) -> anyhow::Result<()> {
    let mut config = load_runtime_config()?;
    config.adjust_for_backend(backend);
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);
    let queries = vera_core::retrieval::normalize_queries(queries);

    if queries.is_empty() {
        bail!(
            "search query is empty.\n\
             Hint: pass at least one non-empty quoted query."
        );
    }

    let (index_dir, filters) =
        prepare_indexed_search(&config.indexing, filters, git_scope.as_ref())?;
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| anyhow::anyhow!("failed to create async runtime: {e}"))?;
    let search_context = rt.block_on(SearchContext::new(&config, backend));

    let runner = SearchRunner {
        rt: &rt,
        search_context: &search_context,
        index_dir: &index_dir,
        config: &config,
        filters: &filters,
        result_limit,
        deep,
    };

    let (results, timings) = if queries.len() == 1 {
        runner.execute_query(&queries[0], intent)?
    } else {
        runner.execute_multi_query_search(&queries, intent)?
    };

    output_results(
        &results,
        json_output,
        raw,
        compact,
        config.retrieval.max_output_chars,
    );

    if timing {
        print_timings(&timings);
    }

    Ok(())
}

struct SearchRunner<'a> {
    rt: &'a tokio::runtime::Runtime,
    search_context: &'a SearchContext,
    index_dir: &'a Path,
    config: &'a VeraConfig,
    filters: &'a SearchFilters,
    result_limit: usize,
    deep: bool,
}

impl SearchRunner<'_> {
    fn execute_query(
        &self,
        query: &str,
        intent: Option<&str>,
    ) -> anyhow::Result<(Vec<SearchResult>, SearchTimings)> {
        if self.deep {
            self.rt.block_on(
                vera_core::retrieval::rag_fusion::execute_deep_search_with_context(
                    self.search_context,
                    self.index_dir,
                    query,
                    intent,
                    self.config,
                    self.filters,
                    self.result_limit,
                ),
            )
        } else {
            self.rt.block_on(self.search_context.search(
                self.index_dir,
                query,
                intent,
                self.config,
                self.filters,
                self.result_limit,
            ))
        }
    }

    fn execute_multi_query_search(
        &self,
        queries: &[String],
        intent: Option<&str>,
    ) -> anyhow::Result<(Vec<SearchResult>, SearchTimings)> {
        let overall_start = Instant::now();
        let per_query_limit = vera_core::retrieval::multi_query_candidate_limit(self.result_limit);
        let mut timings = SearchTimings::default();
        let mut result_sets = Vec::with_capacity(queries.len());

        for query in queries {
            let query_runner = SearchRunner {
                result_limit: per_query_limit,
                ..*self
            };
            let (results, query_timings) = query_runner.execute_query(query, intent)?;
            merge_timings(&mut timings, &query_timings);
            result_sets.push(results);
        }

        let fused = vera_core::retrieval::fuse_and_augment_multi_query(
            self.index_dir,
            queries,
            &result_sets,
            self.filters,
            self.config.retrieval.rrf_k,
            self.result_limit,
            self.result_limit,
        )?;
        timings.total = Some(overall_start.elapsed());
        Ok((fused, timings))
    }
}

fn merge_timings(target: &mut SearchTimings, incoming: &SearchTimings) {
    add_duration(&mut target.embedding, incoming.embedding);
    add_duration(&mut target.bm25, incoming.bm25);
    add_duration(&mut target.vector, incoming.vector);
    add_duration(&mut target.fusion, incoming.fusion);
    add_duration(&mut target.reranking, incoming.reranking);
    add_duration(&mut target.augmentation, incoming.augmentation);
}

fn add_duration(target: &mut Option<Duration>, incoming: Option<Duration>) {
    if let Some(incoming) = incoming {
        *target = Some(target.unwrap_or_default() + incoming);
    }
}

fn print_timings(timings: &SearchTimings) {
    let stderr = std::io::stderr();
    let mut err = stderr.lock();
    let fmt = |d: Option<Duration>| -> String {
        match d {
            Some(d) => format!("{}ms", d.as_millis()),
            None => "n/a".to_string(),
        }
    };
    let stages: &[(&str, Option<Duration>)] = &[
        ("embedding", timings.embedding),
        ("bm25", timings.bm25),
        ("vector", timings.vector),
        ("fusion", timings.fusion),
        ("reranking", timings.reranking),
        ("augmentation", timings.augmentation),
        ("total", timings.total),
    ];
    for (name, duration) in stages {
        if duration.is_some() || *name == "total" {
            let _ = writeln!(err, "[timing] {name}: {}", fmt(*duration));
        }
    }
}
