//! Query-aware ranking heuristics layered on top of dense + lexical retrieval.
//!
//! These heuristics intentionally stay simple and deterministic. They target
//! recurring benchmark failures that single-vector retrieval struggles with:
//! config files at repo root, test/docs noise, symbol-type disambiguation, and
//! same-file crowding for multi-file questions.

use crate::config::VeraConfig;
use crate::corpus::{classify_path, content_class_label};
use crate::types::{Language, SearchFilters, SearchResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RankingStage {
    Initial,
    PostRerank,
}

pub(crate) mod query;
pub(crate) mod score;

#[cfg(test)]
mod tests;

use query::*;
use score::*;

#[cfg(test)]
pub(crate) fn apply_query_ranking(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
) -> Vec<SearchResult> {
    apply_query_ranking_with_filters(query, results, stage, &SearchFilters::default())
}

pub(crate) fn apply_query_ranking_with_filters(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Vec<SearchResult> {
    apply_query_ranking_with_filters_and_config(
        query,
        results,
        stage,
        filters,
        &VeraConfig::default(),
    )
}

pub(crate) fn apply_query_ranking_with_filters_and_config(
    query: &str,
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
    config: &VeraConfig,
) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    let features = QueryFeatures::from_query(query);
    let wants_diversity = features.wants_multi_file_diversity;
    let scores = score_pool_with_config(&features, stage, filters, &results, config);
    finish_ranking(results, scores, wants_diversity)
}

/// Multi-query ranking: each subquery carries its own identifier or filename
/// target, so score the pool under every subquery's features and keep each
/// result's best score. A single joined query would promote only the first
/// subquery's exact match and crowd out the rest (issue #121).
#[allow(dead_code)]
pub(crate) fn apply_query_ranking_multi_query(
    queries: &[String],
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
) -> Vec<SearchResult> {
    apply_query_ranking_multi_query_with_config(
        queries,
        results,
        stage,
        filters,
        &VeraConfig::default(),
    )
}

pub(crate) fn apply_query_ranking_multi_query_with_config(
    queries: &[String],
    results: Vec<SearchResult>,
    stage: RankingStage,
    filters: &SearchFilters,
    config: &VeraConfig,
) -> Vec<SearchResult> {
    if queries.is_empty() || results.len() <= 1 {
        return results;
    }

    let mut scores = vec![f64::NEG_INFINITY; results.len()];
    let mut wants_diversity = false;
    for query in queries {
        let features = QueryFeatures::from_query(query);
        wants_diversity |= features.wants_multi_file_diversity;
        for (best, score) in scores.iter_mut().zip(score_pool_with_config(
            &features, stage, filters, &results, config,
        )) {
            *best = best.max(score);
        }
    }
    finish_ranking(results, scores, wants_diversity)
}

/// Score every pool entry under one query's features: retrieval position
/// (base rank), additive priors, then pool-relative boosts scaled by the
/// pool's best combined score so signal strength tracks retrieval confidence.
#[allow(dead_code)]
fn score_pool(
    features: &QueryFeatures,
    stage: RankingStage,
    filters: &SearchFilters,
    results: &[SearchResult],
) -> Vec<f64> {
    score_pool_with_config(features, stage, filters, results, &VeraConfig::default())
}

fn score_pool_with_config(
    features: &QueryFeatures,
    stage: RankingStage,
    filters: &SearchFilters,
    results: &[SearchResult],
    config: &VeraConfig,
) -> Vec<f64> {
    let retrieval = &config.retrieval;
    let len = results.len() as f64;
    let mut scores: Vec<f64> = results
        .iter()
        .enumerate()
        .map(|(idx, result)| {
            let base_rank = 1.0 - (idx as f64 / len);
            let prior = score_prior_with_config(features, result, stage, filters, retrieval);
            base_rank + prior
        })
        .collect();

    let max_score = scores.iter().copied().fold(0.0_f64, f64::max).max(1e-6);
    apply_coherence_boost(features, &mut scores, results, max_score);
    if retrieval.ranking_filename_stem_boost_enabled() {
        apply_keyword_path_boost(features, &mut scores, results, max_score);
    }
    if retrieval.ranking_definition_boost_enabled() {
        apply_content_symbol_boost(features, &mut scores, results, max_score);
    }
    if retrieval.ranking_multiplicative_path_penalty_enabled() {
        apply_multiplicative_path_penalty(features, &mut scores, results);
    }

    scores
}

fn finish_ranking(
    results: Vec<SearchResult>,
    scores: Vec<f64>,
    wants_diversity: bool,
) -> Vec<SearchResult> {
    let mut scored: Vec<(f64, usize, SearchResult)> = results
        .into_iter()
        .enumerate()
        .zip(scores)
        .map(|((idx, mut result), score)| {
            result.score = score;
            (score, idx, result)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let reranked = scored.into_iter().map(|(_, _, result)| result).collect();
    let reranked = if wants_diversity {
        diversify_by_file(reranked)
    } else {
        reranked
    };
    stamp_rank_scores(reranked)
}

pub(crate) fn file_role_label(file_path: &str, language: Language) -> &'static str {
    content_class_label(classify_path(file_path, language))
}

pub(crate) fn is_path_weighted_query(query: &str) -> bool {
    let lower = query.trim().to_ascii_lowercase();
    if lower.contains(".toml")
        || lower.contains(".json")
        || lower.contains(".yaml")
        || lower.contains(".yml")
        || lower.contains(".ini")
        || lower.contains(".conf")
        || lower.contains("dockerfile")
        || lower.contains("makefile")
        || lower.contains("cmakelists.txt")
    {
        return true;
    }

    // A slash alone does not make a path query: prose like "read/write
    // request handling" must stay semantic. Require a slash-bearing token
    // that is the whole query or has path shape (prefix or file extension).
    let tokens: Vec<&str> = lower
        .split_whitespace()
        .map(crate::retrieval::query_utils::trim_query_token)
        .filter(|token| !token.is_empty())
        .collect();
    tokens
        .iter()
        .any(|token| is_path_shaped_token(token, tokens.len() == 1))
}

fn is_path_shaped_token(token: &str, single_token_query: bool) -> bool {
    if !token.contains('/') && !token.contains('\\') {
        return false;
    }
    if single_token_query
        || token.starts_with("./")
        || token.starts_with("../")
        || token.starts_with('/')
        || token.starts_with('~')
        || token.as_bytes().get(1) == Some(&b':')
    // Windows drive prefix
    {
        return true;
    }
    // src/main.rs: the last path segment carries a file extension.
    let last_segment = token.rsplit(['/', '\\']).next().unwrap_or(token);
    last_segment.contains('.')
}
