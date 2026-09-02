//! Prior scoring and result-shaping heuristics.

use crate::chunk_text::file_name;
use crate::config::{RetrievalConfig, VeraConfig};
use crate::corpus::{ContentClass, classify_content};
use crate::retrieval::query_classifier::QueryType;
use crate::retrieval::query_utils::{
    content_declares_public_symbol, content_starts_with_impl, file_stem, path_depth,
    trim_query_token,
};
use crate::types::{SearchFilters, SearchResult, SymbolType};
use std::collections::HashMap;

use super::query::*;
use super::*;

// Ranking signal weights and thresholds for deterministic result shaping.
/// Per-file score-sum coherence, boosting only the file's best chunk.
pub(super) const COHERENCE_WEIGHT: f64 = 0.4;
/// Coherence weight for natural-language queries.
pub(super) const COHERENCE_WEIGHT_NL: f64 = 0.4;
/// Pool-relative filename and parent-directory keyword boost scale.
pub(super) const KEYWORD_PATH_WEIGHT: f64 = 1.0;
/// Minimum coverage ratio required before applying the content boost.
pub(super) const COVERAGE_MIN_RATIO: f64 = 0.5;
/// Content coverage boost weight.
pub(super) const COVERAGE_WEIGHT: f64 = 2.4;
/// Coverage weight for multi-word identifier queries.
pub(super) const COVERAGE_WEIGHT_IDENT: f64 = 2.0;
/// Coverage curve exponent. Values above 1 damp partial coverage while
/// preserving the full-coverage signal.
pub(super) const COVERAGE_EXPONENT: f64 = 2.0;
/// Lower coverage weight for multi-facet queries with explicit conjunctions.
pub(super) const COVERAGE_WEIGHT_CONJ: f64 = 1.6;

/// Coverage weight for the query: strongest for single-topic NL questions,
/// gentler when the query is multi-facet (explicit conjunction) or names a
/// symbol (symbol-definition signals should dominate there), and for
/// Identifier queries which have symbol-specific signals already.
fn coverage_weight(features: &QueryFeatures) -> f64 {
    if features.query_type != QueryType::NaturalLanguage {
        return COVERAGE_WEIGHT_IDENT;
    }
    if features.has_conjunction || !features.embedded_symbols.is_empty() {
        COVERAGE_WEIGHT_CONJ
    } else {
        COVERAGE_WEIGHT
    }
}

/// Coverage of query content words in the chunk. Returns the covered
/// fraction when the signal applies and clears the minimum ratio.
fn coverage_ratio(features: &QueryFeatures, content: &str) -> Option<f64> {
    let coverage_keywords: Vec<&String> = features.raw_keywords.iter().collect();
    if coverage_keywords.len() < 2 {
        return None;
    }
    let content_lower = content.to_ascii_lowercase();
    let covered = coverage_keywords
        .iter()
        .filter(|kw| content_covers_keyword(&content_lower, kw))
        .count();
    let ratio = covered as f64 / coverage_keywords.len() as f64;
    (ratio >= COVERAGE_MIN_RATIO).then_some(ratio)
}

/// Inflection-tolerant coverage. Queries inflect verbs ("parsing")
/// while code prose uses other forms ("parses"); stripping a trailing
/// "ing"/"ed"/"s" (4+ char stem kept) recovers those matches.
fn content_covers_keyword(content_lower: &str, keyword: &str) -> bool {
    if content_lower.contains(keyword) {
        return true;
    }
    for suffix in ["ing", "ed", "s"] {
        if let Some(stem) = keyword.strip_suffix(suffix)
            && stem.len() >= 4
            && content_lower.contains(stem)
        {
            return true;
        }
    }
    false
}

/// Embedded-symbol content-definition weight for natural-language queries.
pub(super) const CONTENT_SYMBOL_WEIGHT_EMBEDDED: f64 = 1.5;
/// Content-definition weight for identifier queries.
pub(super) const CONTENT_SYMBOL_WEIGHT_IDENT: f64 = 3.0;

#[allow(dead_code)]
pub(super) fn score_prior(
    features: &QueryFeatures,
    result: &SearchResult,
    stage: RankingStage,
    filters: &SearchFilters,
) -> f64 {
    score_prior_with_config(
        features,
        result,
        stage,
        filters,
        &VeraConfig::default().retrieval,
    )
}

pub(super) fn score_prior_with_config(
    features: &QueryFeatures,
    result: &SearchResult,
    stage: RankingStage,
    filters: &SearchFilters,
    retrieval_config: &RetrievalConfig,
) -> f64 {
    let stage_weight = match stage {
        RankingStage::Initial => 1.0,
        RankingStage::PostRerank => 0.55,
    };
    let depth = path_depth(&result.file_path) as f64;
    let role = classify_content(&result.file_path, result.language, &result.content);
    let mut bonus = 0.0;
    let file_path = result.file_path.to_ascii_lowercase();
    let result_filename = file_name(&result.file_path).to_ascii_lowercase();
    let allow_filename_semantic_bonus = matches!(
        role,
        ContentClass::Source | ContentClass::Config | ContentClass::Unknown
    );
    let path_fragment_match = features
        .path_fragment
        .as_deref()
        .is_some_and(|fragment| path_matches_fragment(&file_path, fragment));
    let filename_boost_allowed = features.path_fragment.is_none() || path_fragment_match;

    if path_fragment_match {
        bonus += stage_weight * 1.2;
    }

    if let Some(filename) = features.exact_filename.as_deref() {
        if filename_boost_allowed && result_filename == filename {
            let filename_bonus = if features.wants_config_paths {
                if depth == 0.0 {
                    1.15
                } else {
                    (0.45 - depth.min(5.0) * 0.08).max(0.08)
                }
            } else if depth == 0.0 {
                0.9
            } else {
                (0.6 - depth.min(5.0) * 0.06).max(0.12)
            };
            bonus += stage_weight * filename_bonus;
        } else if filename_boost_allowed && file_path.ends_with(filename) {
            bonus += stage_weight * 0.15;
        }
    }

    if features.wants_config_paths && matches!(role, ContentClass::Config) {
        bonus += stage_weight
            * if depth == 0.0 {
                0.35
            } else {
                (0.2 - depth.min(5.0) * 0.03).max(0.05)
            };
    }

    if let Some(identifier) = features.exact_identifier.as_deref() {
        if result
            .symbol_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(identifier))
        {
            // Symbol name matches the query identifier. This is the strongest
            // signal: developers searching "Axios" want the Axios class definition.
            let is_definition_chunk = is_definition_symbol(result.symbol_type);
            let stem_aligns = file_stem(&result_filename).eq_ignore_ascii_case(identifier);
            let base_symbol_bonus = if features.query_word_count <= 2 {
                if is_definition_chunk { 1.6 } else { 0.7 }
            } else if is_definition_chunk {
                1.2
            } else {
                0.55
            };
            bonus += stage_weight * base_symbol_bonus;
            bonus += stage_weight * if depth <= 2.0 { 0.18 } else { 0.05 };
            if features.requested_symbol_types.contains(&SymbolType::Class)
                && is_internal_definition_path(&file_path)
            {
                bonus -= stage_weight * 0.35;
            }
            if features
                .exact_identifier_case
                .as_deref()
                .is_some_and(|name| result.symbol_name.as_deref() == Some(name))
            {
                bonus += stage_weight * 0.28;
            }
            // Extra boost when the file stem also matches (e.g., Axios in Axios.js).
            if stem_aligns {
                bonus += stage_weight * 0.45;
            }
        } else if file_stem(&result_filename).eq_ignore_ascii_case(identifier) {
            bonus += stage_weight * 0.35;
        } else if file_stem_prefix_matches_identifier(file_stem(&result_filename), identifier) {
            bonus += stage_weight * 0.28;
        } else if identifier_matches_parent_dir(identifier, &file_path) {
            bonus += stage_weight * 0.22;
        }
    }

    if features.query_type == QueryType::NaturalLanguage
        && !features.keywords.is_empty()
        && !features.wants_config_paths
        && allow_filename_semantic_bonus
        && let Some(symbol_name) = result.symbol_name.as_deref()
    {
        let symbol_bonus = symbol_keyword_bonus(symbol_name, &features.keywords);
        if symbol_bonus > 0.0 {
            bonus += stage_weight * symbol_bonus;
        }
    }

    if !features.requested_symbol_types.is_empty()
        && result
            .symbol_type
            .is_some_and(|sym| features.requested_symbol_types.contains(&sym))
    {
        bonus += stage_weight * 0.62;
        if features
            .exact_identifier_case
            .as_deref()
            .is_some_and(|name| result.symbol_name.as_deref() == Some(name))
        {
            bonus += stage_weight * 0.2;
        }
    } else if !features.requested_symbol_types.is_empty() {
        bonus -= stage_weight
            * if features.exact_identifier_case.is_some() {
                0.9
            } else {
                0.55
            };
    }

    if features.mentions_definition && is_definition_symbol(result.symbol_type) {
        bonus += stage_weight
            * if result.symbol_name.is_some() {
                0.34
            } else {
                0.18
            };
    }

    // Boost definition chunks for NL queries when their symbol name overlaps
    // query keywords. Definitions are the canonical location for a concept;
    // they should strongly outrank incidental mentions. Use a weaker boost
    // for broad multi-keyword queries where the symbol match is partial.
    // Gated by ranking_definition_boost (issue #196 signal).
    if retrieval_config.ranking_definition_boost_enabled()
        && features.query_type == QueryType::NaturalLanguage
        && is_definition_symbol(result.symbol_type)
        && result.symbol_name.is_some()
        && let Some(symbol_name) = result.symbol_name.as_deref()
    {
        let sym_stems = identifier_stems(symbol_name);
        // Count keyword overlaps where the keyword is non-trivial (5+ chars)
        // to avoid short keywords like "file", "type", "list" causing false boosts.
        let overlap_count = features
            .keywords
            .iter()
            .filter(|kw| {
                kw.len() >= 5
                    && sym_stems
                        .iter()
                        .any(|s| s == kw.as_str() || shares_keyword_stem(s, kw))
            })
            .count();
        if overlap_count > 0 {
            // Scale by overlap ratio: single keyword match in a 5-word query
            // gets a modest boost; full overlap gets the maximum.
            let long_keywords = features
                .keywords
                .iter()
                .filter(|k| k.len() >= 5)
                .count()
                .max(1);
            let ratio = (overlap_count as f64 / long_keywords as f64).min(1.0);

            // Extra boost when the file stem also matches the symbol.
            let stem = file_stem(&result_filename);
            let stem_aligns = file_stem(&result_filename).eq_ignore_ascii_case(symbol_name)
                || sym_stems.iter().any(|s| {
                    s == &normalize_token(stem) || shares_keyword_stem(s, &normalize_token(stem))
                });
            let base_boost = if stem_aligns { 1.5 } else { 1.0 };
            bonus += stage_weight * base_boost * ratio;
        }
    }

    // Content-based definition detection: if the chunk's content defines
    // a symbol matching query keywords (via language-agnostic prefix matching),
    // boost it. This catches cases where symbol_type metadata is missing
    // or too coarse. Skip when the user wants non-source content.
    // Use a mild boost; the metadata-based definition boost above handles
    // strong signals. Gated by ranking_definition_boost.
    if retrieval_config.ranking_definition_boost_enabled()
        && features.query_type == QueryType::NaturalLanguage
        && !features.keywords.is_empty()
        && !features.wants_runtime_paths
        && !features.wants_config_paths
        && content_defines_query_keyword(&result.content, &features.keywords)
    {
        bonus += stage_weight * 0.3;
    }

    // Content keyword coverage: multi-concept NL questions are answered by
    // the chunk that mentions every concept, and BM25's frequency saturation
    // can bury such a chunk under single-concept keyword-dense noise. Reward
    // the fraction of distinct query keywords the chunk content covers.
    // Explicit config-path requests use path intent instead of content
    // coverage.
    if !features.wants_config_paths
        && let Some(ratio) = coverage_ratio(features, &result.content)
    {
        bonus += stage_weight * coverage_weight(features) * ratio.powf(COVERAGE_EXPONENT);
    }

    // --- Noise penalties ---
    if !features.wants_test_paths && matches!(role, ContentClass::Test) {
        bonus -= stage_weight * 0.95;
    }
    if matches!(role, ContentClass::Archive) {
        if features.wants_archive_paths {
            bonus += stage_weight * 0.18;
        } else {
            bonus -= stage_weight * 0.85;
        }
    }
    if matches!(role, ContentClass::Runtime) {
        if features.wants_runtime_paths {
            bonus += stage_weight * 0.95;
        } else {
            bonus -= stage_weight * 0.72;
        }
    } else if features.wants_runtime_paths {
        bonus -= stage_weight * 0.24;
    }
    if !features.wants_docs_paths && matches!(role, ContentClass::Docs) {
        bonus -= stage_weight
            * if prefers_source_over_docs(features) {
                0.95
            } else {
                0.55
            };
    }
    if !features.wants_example_paths && matches!(role, ContentClass::Example | ContentClass::Bench)
    {
        bonus -= stage_weight * 0.55;
    }
    if !features.wants_compat_paths && is_compat_path(&file_path) {
        bonus -= stage_weight * 0.65;
    } else if features.wants_compat_paths && is_compat_path(&file_path) {
        bonus += stage_weight * 0.32;
    }
    if !features.wants_type_declarations && is_typescript_declaration(&file_path) {
        bonus -= stage_weight * 0.82;
    }
    if is_reexport_barrel(result) && !features.mentions_definition {
        bonus -= stage_weight * 0.95;
    }
    bonus += stage_weight * version_path_bonus(features, &file_path);
    if matches!(role, ContentClass::Generated) {
        bonus -= stage_weight
            * if features.wants_runtime_paths {
                0.18
            } else {
                0.95
            };
        if filters.include_generated == Some(false) {
            bonus -= stage_weight * 0.8;
        }
    }
    if matches!(role, ContentClass::Source | ContentClass::Config) {
        bonus += stage_weight
            * if features.query_type == QueryType::Identifier || features.path_fragment.is_some() {
                if depth <= 2.0 { 0.24 } else { 0.12 }
            } else if depth <= 2.0 {
                0.12
            } else {
                0.05
            };
    }
    if let Some(scope) = filters.scope {
        if crate::corpus::matches_scope(role, scope, filters.include_generated.unwrap_or(true)) {
            bonus += stage_weight * 0.18;
        } else {
            bonus -= stage_weight * 1.1;
        }
    }

    if features.mentions_implementation && looks_like_impl_block(result) {
        bonus += stage_weight * 0.18;
    }

    if features.query_type == QueryType::NaturalLanguage && is_public_symbol(result) {
        bonus += stage_weight * 0.05;
    }

    if prefers_structural_chunks(features) {
        bonus += stage_weight * structural_chunk_bias(result);
    }

    bonus
}

pub(super) fn prefers_source_over_docs(features: &QueryFeatures) -> bool {
    features.query_type == QueryType::NaturalLanguage
        && features.query_word_count >= 4
        && !features.wants_config_paths
        && !features.wants_runtime_paths
        && !features.wants_archive_paths
}

pub(super) fn requested_symbol_types(query: &str) -> Vec<SymbolType> {
    let mut symbol_types = Vec::new();
    if query.contains("trait") {
        symbol_types.push(SymbolType::Trait);
    }
    if query.contains("class") {
        symbol_types.push(SymbolType::Class);
    }
    if query.contains("interface") {
        symbol_types.push(SymbolType::Interface);
    }
    if query.contains("struct") {
        symbol_types.push(SymbolType::Struct);
    }
    if query.contains("enum") {
        symbol_types.push(SymbolType::Enum);
    }
    if query.contains("function") {
        symbol_types.push(SymbolType::Function);
    }
    if query.contains("method") {
        symbol_types.push(SymbolType::Method);
    }
    symbol_types
}

pub(super) fn requested_versions(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|token| {
            token.len() >= 2
                && token.starts_with('v')
                && token[1..].chars().all(|ch| ch.is_ascii_digit())
        })
        .cloned()
        .collect()
}

/// Maximum chunks from the same file before saturation decay kicks in.
pub(super) const FILE_SATURATION_THRESHOLD: usize = 1;

/// Multiplicative penalty per extra chunk from the same file beyond the threshold.
/// 0.35 means each successive same-file chunk keeps 35% of its score, pushing
/// it below results from other files in most cases.
pub(super) const FILE_SATURATION_DECAY: f64 = 0.35;

/// File coherence: files whose chunks collectively score well are
/// likely the file the user needs. Sum each file's (clamped) combined scores
/// and boost only the file's best chunk, proportional to the file's share of
/// the strongest file. Boosting every chunk (count-based) lets one file flood
/// the window; boosting only the best surfaces the cluster without the flood.
pub(super) fn apply_coherence_boost(
    features: &QueryFeatures,
    scores: &mut [f64],
    results: &[SearchResult],
    max_score: f64,
) {
    // Identifier queries benefit from a stronger coherence weight; for
    // natural-language questions it costs intent ordering.
    let weight = if features.query_type == QueryType::NaturalLanguage {
        COHERENCE_WEIGHT_NL
    } else {
        COHERENCE_WEIGHT
    };
    let updates: Vec<(usize, f64)> = {
        let mut file_sum: HashMap<&str, f64> = HashMap::new();
        for (score, result) in scores.iter().zip(results) {
            *file_sum.entry(result.file_path.as_str()).or_default() += score.max(0.0);
        }
        let max_file_sum = file_sum.values().copied().fold(0.0_f64, f64::max).max(1e-6);

        let mut best_chunk: HashMap<&str, usize> = HashMap::new();
        for (i, (score, result)) in scores.iter().zip(results).enumerate() {
            best_chunk
                .entry(result.file_path.as_str())
                .and_modify(|j| {
                    if *score > scores[*j] {
                        *j = i;
                    }
                })
                .or_insert(i);
        }

        best_chunk
            .into_iter()
            .map(|(path, i)| {
                let boost = weight
                    * max_score
                    * (file_sum.get(path).copied().unwrap_or(0.0) / max_file_sum);
                (i, boost)
            })
            .collect()
    };

    for (i, boost) in updates {
        scores[i] += boost;
    }
}

/// Pool-relative keyword path boost: when query keywords match a file's stem
/// or its immediate parent directory (exact or prefix, 3+ chars), every chunk
/// of that file gains `max_score * match_ratio`. Scaling by the pool's best
/// score keeps the signal proportional to retrieval confidence, and the
/// match-ratio form rewards files named after the whole query over files
/// matching a single incidental keyword.
pub(super) fn apply_keyword_path_boost(
    features: &QueryFeatures,
    scores: &mut [f64],
    results: &[SearchResult],
    max_score: f64,
    retrieval: &RetrievalConfig,
) {
    // Explicit runtime intent overrides filename keyword inference: a user
    // asking for the runtime extract does not want the like-named source file
    // boosted past it. (Other intent flags are not gated: e.g. "compat" is
    // both a content-class flag and a legitimate path keyword, and the
    // role gate below already excludes non-source files from the boost.)
    if features.query_type != QueryType::NaturalLanguage
        || features.wants_config_paths
        || features.wants_runtime_paths
    {
        return;
    }
    // Gating knob: skip symbol queries when the exact-identifier machinery
    // is engaged. This suppresses the filename inference for symbol lookups
    // which already have dedicated boosts.
    if retrieval.ranking_filename_stem_skip_symbol_queries_enabled()
        && (features.exact_identifier.is_some() || !features.embedded_symbols.is_empty())
    {
        return;
    }
    if features.keywords.is_empty() {
        return;
    }
    let keywords: Vec<&str> = features
        .keywords
        .iter()
        .map(String::as_str)
        .filter(|kw| kw.len() > 2)
        .collect();
    if keywords.is_empty() {
        return;
    }

    let min_ratio = retrieval.ranking_filename_stem_min_ratio_effective();
    let mut bonuses = Vec::with_capacity(results.len());
    {
        let mut bonus_cache: HashMap<&str, f64> = HashMap::new();
        for result in results {
            let bonus = *bonus_cache
                .entry(result.file_path.as_str())
                .or_insert_with(|| {
                    let role =
                        classify_content(&result.file_path, result.language, &result.content);
                    if !matches!(
                        role,
                        ContentClass::Source | ContentClass::Config | ContentClass::Unknown
                    ) {
                        return 0.0;
                    }
                    let ratio = keyword_path_match_ratio(&keywords, &result.file_path);
                    if ratio >= min_ratio {
                        KEYWORD_PATH_WEIGHT * max_score * ratio
                    } else {
                        0.0
                    }
                });
            bonuses.push(bonus);
        }
    }
    for (score, bonus) in scores.iter_mut().zip(bonuses) {
        *score += bonus;
    }
}

/// Backward-compatible wrapper for tests that don't pass config (defaults preserved).
#[allow(dead_code)]
pub(super) fn apply_keyword_path_boost_legacy(
    features: &QueryFeatures,
    scores: &mut [f64],
    results: &[SearchResult],
    max_score: f64,
) {
    apply_keyword_path_boost(
        features,
        scores,
        results,
        max_score,
        &RetrievalConfig::default(),
    )
}

/// Pool-relative content-based symbol definition boost. A chunk whose
/// text actually defines the queried symbol ("class Session",
/// "CREATE TABLE sessions") is the definition site regardless of what symbol
/// metadata extraction produced. Symbol queries scale the boost by
/// 3.0 * pool max; embedded symbols in NL queries get half strength (the
/// symbol may be incidental to the question). The file named after the
/// symbol earns the 1.5x stem-aligned tier.
pub(super) fn apply_content_symbol_boost(
    features: &QueryFeatures,
    scores: &mut [f64],
    results: &[SearchResult],
    max_score: f64,
) {
    let targets: Vec<(String, f64)> = match features.query_type {
        QueryType::Identifier => {
            let Some(name) = features.exact_identifier_case.as_deref() else {
                return;
            };
            // Match the final segment of qualified names ("std::io::Error" ->
            // "Error"); keep the full form as a fallback name.
            let short = name
                .rsplit([':', '\\', '.'])
                .next()
                .unwrap_or(name)
                .to_string();
            let mut targets = vec![(short, CONTENT_SYMBOL_WEIGHT_IDENT)];
            if !targets[0].0.eq_ignore_ascii_case(name) {
                targets.push((name.to_string(), CONTENT_SYMBOL_WEIGHT_IDENT));
            }
            targets
        }
        QueryType::NaturalLanguage => {
            if features.embedded_symbols.is_empty() {
                return;
            }
            features
                .embedded_symbols
                .iter()
                .map(|symbol| (symbol.clone(), CONTENT_SYMBOL_WEIGHT_EMBEDDED))
                .collect()
        }
    };

    let identifier_query = features.query_type == QueryType::Identifier;
    for (score, result) in scores.iter_mut().zip(results) {
        // A definition inside test/example content is a fixture or usage
        // sample, not the definition site a symbol query is after.
        if definition_site_role_blocked(features, result) {
            continue;
        }
        let mut boost = 0.0_f64;
        for (name, unit) in &targets {
            // Backstop only: chunks whose symbol metadata already matches the
            // identifier are handled by the metadata symbol boost. The content
            // scan covers extraction gaps (SQL DDL, multi-symbol chunks).
            if identifier_query
                && result
                    .symbol_name
                    .as_deref()
                    .is_some_and(|s| s.eq_ignore_ascii_case(name))
            {
                continue;
            }
            if content_defines_symbol(&result.content, name) {
                let filename = file_name(&result.file_path).to_ascii_lowercase();
                let stem = file_stem(&filename);
                let stem_matched = stem_matches_symbol(stem, name);
                // Identifier lookups get a stronger stem-aligned tier, while
                // unrestricted content matching still covers extraction gaps.
                let tier = if stem_matched { 1.5 } else { 1.0 };
                boost = boost.max(unit * max_score * tier);
            }
        }
        *score += boost;
    }
}

/// Multiplicative path penalty factor for test/compat/example directories.
/// ~0.3× as hypothesized in #196: boilerplate / fixture / example directories
/// are keyword-dense but rarely contain the implementation sought. Multiplying
/// (rather than subtracting) demotes proportionally to retrieval confidence,
/// preserving ordering among non-penalized candidates while consistently
/// demoting penalized ones.
pub(super) const MULTIPLICATIVE_PATH_PENALTY_FACTOR: f64 = 0.3;

/// Shared directory-classification constants (single definition to prevent drift).
const TEST_DIRS: &[&str] = &[
    "t",
    "test",
    "tests",
    "testing",
    "__tests__",
    "spec",
    "specs",
    "testdata",
    "fixture",
    "fixtures",
];
const EXAMPLE_DIRS: &[&str] = &[
    "example",
    "examples",
    "sample",
    "samples",
    "demo",
    "demos",
    "bench",
    "benches",
    "benchmark",
    "benchmarks",
];

/// Apply the ~0.3× multiplicative penalty for test/compat/example paths.
///
/// Respects the existing boost-directory gating (definition boost's
/// `wants_*` checks) so explicit requests for those paths are not penalized.
/// A file is penalized if its path lies in `tests/`, `compat/`, or
/// `examples/` (or is classified as `Test`/`Example`/`Bench`) and the query
/// does not ask for that category. The penalty multiplies the score, so two
/// equally-scoring candidates (`src/` vs `tests/`) will rank `src/` higher
/// when the knob is on.
pub(super) fn apply_multiplicative_path_penalty(
    features: &QueryFeatures,
    scores: &mut [f64],
    results: &[SearchResult],
) {
    for (score, result) in scores.iter_mut().zip(results) {
        let lower = result.file_path.to_ascii_lowercase();
        let role = classify_content(&result.file_path, result.language, &result.content);
        let mut penalized = false;

        // Test / fixture penalization: mirrors the additive penalty's
        // ContentClass gate plus the definition-boost directory gating so
        // both signals respect the same explicit-request logic.
        if !features.wants_test_paths
            && (matches!(role, ContentClass::Test)
                || definition_site_role_blocked_is_test(features, result))
        {
            penalized = true;
        }
        if !penalized
            && !features.wants_example_paths
            && (matches!(role, ContentClass::Example | ContentClass::Bench)
                || definition_site_role_blocked_is_example(features, result))
        {
            penalized = true;
        }
        if !penalized && !features.wants_compat_paths && is_compat_path(&lower) {
            penalized = true;
        }

        if penalized {
            *score *= MULTIPLICATIVE_PATH_PENALTY_FACTOR;
        }
    }
}

fn definition_site_role_blocked_is_test(features: &QueryFeatures, result: &SearchResult) -> bool {
    let lower = result.file_path.to_ascii_lowercase();
    let mut parts = lower.rsplit('/');
    let filename = parts.next().unwrap_or("");
    let in_test_dir = parts.any(|dir| TEST_DIRS.contains(&dir));
    if in_test_dir || is_test_filename(filename) {
        return !features.wants_test_paths;
    }
    false
}

fn definition_site_role_blocked_is_example(
    features: &QueryFeatures,
    result: &SearchResult,
) -> bool {
    let lower = result.file_path.to_ascii_lowercase();
    let mut parts = lower.rsplit('/');
    let filename = parts.next().unwrap_or("");
    let in_example_dir = parts.clone().any(|dir| EXAMPLE_DIRS.contains(&dir));
    // is_test_filename part already handled in test check; example only cares about dir
    if in_example_dir {
        return !features.wants_example_paths;
    }
    // also treat example-like filenames conservatively? filename alone not penalized
    let _ = filename;
    false
}

/// A chunk counts as a definition site for the content-symbol boost only
/// when its path marks it as source-like. Definitions in test, example, and
/// bench trees are fixtures or usage samples; they qualify only when the
/// query explicitly asks for those paths.
///
/// Directory components decide, not bare filename tokens: a first-class
/// module named `testing.py` (click's CliRunner lives in
/// src/click/testing.py) or `example.py` is still a definition site.
fn definition_site_role_blocked(features: &QueryFeatures, result: &SearchResult) -> bool {
    let lower = result.file_path.to_ascii_lowercase();
    let mut parts = lower.rsplit('/');
    let filename = parts.next().unwrap_or("");
    let in_test_dir = parts.clone().any(|dir| TEST_DIRS.contains(&dir));
    let in_example_dir = parts.any(|dir| EXAMPLE_DIRS.contains(&dir));
    if in_test_dir || is_test_filename(filename) {
        return !features.wants_test_paths;
    }
    if in_example_dir {
        return !features.wants_example_paths;
    }
    false
}

/// Conventional test-file names: test_foo.py, foo_test.go, foo.test.ts,
/// foo-spec.rb. Deliberately narrower than substring matching so modules
/// like `testing.py` or `attest.py` stay source-like.
fn is_test_filename(filename: &str) -> bool {
    filename.starts_with("test_")
        || filename.starts_with("test-")
        || filename.starts_with("spec_")
        || filename.starts_with("spec-")
        || filename.contains("_test.")
        || filename.contains("-test.")
        || filename.contains(".test.")
        || filename.contains("_spec.")
        || filename.contains("-spec.")
        || filename.contains(".spec.")
}

/// Stem-vs-symbol match: exact, underscore-normalised, or plural-adjusted.
fn stem_matches_symbol(stem: &str, name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let stem_norm = stem.replace('_', "");
    stem == name
        || stem_norm == name
        || stem.trim_end_matches('s') == name
        || stem_norm.trim_end_matches('s') == name
}

/// General definition keywords, case-sensitive. Matching is done on the
/// original line: code keywords are lowercase in practice, and
/// case-insensitive matching misfires on prose like "Class" or "Module".
const DEF_KEYWORDS: &[&str] = &[
    "abstract class",
    "data class",
    "class",
    "module",
    "defmodule",
    "def",
    "interface",
    "struct",
    "enum",
    "trait",
    "type",
    "func",
    "function",
    "object",
    "fn",
    "fun",
    "package",
    "namespace",
    "protocol",
    "record",
    "typedef",
];

/// SQL DDL definition keywords, matched case-insensitively.
const SQL_DEF_KEYWORDS: &[&str] = &[
    "create table",
    "create view",
    "create procedure",
    "create function",
];

/// Does the chunk content define `symbol`? General keywords match
/// case-sensitively, SQL DDL case-insensitively (conventionally uppercase).
fn content_defines_symbol(content: &str, symbol: &str) -> bool {
    if symbol.len() < 2 {
        return false;
    }
    // Quick reject: the symbol must appear in the chunk at all.
    let symbol_lower;
    if !content.contains(symbol) {
        symbol_lower = symbol.to_ascii_lowercase();
        if !content.to_ascii_lowercase().contains(&symbol_lower) {
            return false;
        }
    }
    let symbol_lower = symbol.to_ascii_lowercase();
    for line in content.lines() {
        if line_defines_symbol(line, symbol, false, DEF_KEYWORDS) {
            return true;
        }
        if line.to_ascii_lowercase().contains(&symbol_lower)
            && line_defines_symbol(
                &line.to_ascii_lowercase(),
                &symbol_lower,
                true,
                SQL_DEF_KEYWORDS,
            )
        {
            return true;
        }
    }
    false
}

/// Scan a line for `keyword symbol` definition sites, e.g. "class Session"
/// or "defmodule Phoenix.Router". Keyword must start the line or follow
/// whitespace.
fn line_defines_symbol(
    line: &str,
    symbol: &str,
    case_insensitive: bool,
    keywords: &[&str],
) -> bool {
    let mut start = 0;
    while start < line.len() {
        let mut earliest: Option<(usize, &str)> = None;
        for keyword in keywords {
            let mut from = start;
            while let Some(pos) = line[from..].find(keyword) {
                let abs = from + pos;
                let left_ok = abs == 0 || line.as_bytes()[abs - 1].is_ascii_whitespace();
                if left_ok {
                    earliest = Some(match earliest {
                        Some((prev, kw)) if prev <= abs => (prev, kw),
                        _ => (abs, keyword),
                    });
                    break;
                }
                from = abs + 1;
            }
        }
        let Some((pos, keyword)) = earliest else {
            return false;
        };
        if rest_starts_with_symbol(&line[pos + keyword.len()..], symbol, case_insensitive) {
            return true;
        }
        start = pos + keyword.len();
    }
    false
}

/// Check whether the text after a definition keyword names `symbol`,
/// skipping namespace qualifiers ("defmodule Phoenix.Router" defines Router).
/// The symbol must end at a delimiter so "Session" does not match
/// "SessionStore".
fn rest_starts_with_symbol(rest: &str, symbol: &str, case_insensitive: bool) -> bool {
    let mut text = rest.trim_start();
    loop {
        let ident_len = text
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .count();
        if ident_len == 0 {
            return false;
        }
        let (ident, after) = text.split_at(ident_len);
        if let Some(stripped) = after.strip_prefix("::") {
            text = stripped.trim_start();
            continue;
        }
        if let Some(stripped) = after.strip_prefix('.')
            && stripped
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            text = stripped.trim_start();
            continue;
        }
        let matches = if case_insensitive {
            ident.eq_ignore_ascii_case(symbol)
        } else {
            ident == symbol
        };
        return matches
            && (after.is_empty()
                || after
                    .chars()
                    .next()
                    .is_some_and(|c| matches!(c, ' ' | '\t' | '<' | '(' | '{' | ':' | '[' | ';')));
    }
}

/// Fraction of query keywords that match the file stem's or immediate parent
/// directory's sub-tokens (exact, or prefix with a 3-char minimum).
pub(crate) fn keyword_path_match_ratio(keywords: &[&str], file_path: &str) -> f64 {
    if keywords.is_empty() {
        return 0.0;
    }
    let lowered = file_path.to_ascii_lowercase();
    let stem = file_stem(file_name(&lowered));
    let mut parts = identifier_stems(stem);
    if let Some((dirs, _)) = lowered.rsplit_once('/')
        && let Some(parent) = dirs.rsplit('/').next()
    {
        parts.extend(identifier_stems(parent));
    }
    if parts.is_empty() {
        return 0.0;
    }

    let matched = keywords
        .iter()
        .filter(|kw| {
            parts.iter().any(|part| {
                part == *kw
                    || (kw.len() >= 3 && part.starts_with(*kw))
                    || (part.len() >= 3 && kw.starts_with(part.as_str()))
            })
        })
        .count();

    (matched as f64 / keywords.len() as f64).min(1.0)
}

pub(super) fn diversify_by_file(results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    use std::collections::HashMap;

    let mut file_counts: HashMap<String, usize> = HashMap::new();
    let mut scored: Vec<(f64, usize, SearchResult)> = results
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            let count = file_counts.entry(result.file_path.clone()).or_insert(0);
            *count += 1;
            let effective_score = if *count > FILE_SATURATION_THRESHOLD {
                let excess = (*count - FILE_SATURATION_THRESHOLD) as f64;
                result.score * FILE_SATURATION_DECAY.powf(excess)
            } else {
                result.score
            };
            (effective_score, idx, result)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    scored.into_iter().map(|(_, _, result)| result).collect()
}

pub(super) fn stamp_rank_scores(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    let len = results.len().max(1) as f64;
    for (idx, result) in results.iter_mut().enumerate() {
        result.score = 1.0 - (idx as f64 / len);
    }
    results
}

pub(super) fn looks_like_path_fragment(token: &str) -> bool {
    token.contains('/') || token.contains('\\')
}

pub(super) fn clean_query_token(token: &str) -> String {
    trim_query_token(token).to_ascii_lowercase()
}

pub(super) fn mentions_any(query: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| query.contains(needle))
}

pub(super) fn is_query_stopword(token: &str) -> bool {
    matches!(
        token,
        "and"
            | "or"
            | "the"
            | "a"
            | "an"
            | "of"
            | "in"
            | "to"
            | "for"
            | "with"
            | "across"
            | "where"
            | "definition"
            | "definitions"
            | "configured"
            | "configuration"
    )
}

pub(super) fn normalize_token(token: &str) -> String {
    let token = token.to_ascii_lowercase();
    let trimmed = token.trim_end_matches('s');
    if trimmed.len() >= 3 {
        trimmed.to_string()
    } else {
        token
    }
}

pub(super) fn tokenize_path(path: &str) -> Vec<&str> {
    path.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|part| !part.is_empty())
        .collect()
}

pub(super) fn contains_token(tokens: &[&str], expected: &[&str]) -> bool {
    tokens.iter().any(|token| expected.contains(token))
}

pub(super) fn is_internal_definition_path(path: &str) -> bool {
    let tokens = tokenize_path(path);
    contains_token(&tokens, &["sansio", "internal", "bindings"])
}

pub(super) fn path_matches_fragment(path: &str, fragment: &str) -> bool {
    path == fragment || path.ends_with(fragment) || path.contains(fragment)
}

/// Check if a file stem shares a 6+ char prefix with an identifier.
/// Strips namespace prefixes (e.g. "sinatra::showexceptions" → "showexceptions")
/// so that "format" matches "formatter" but "sinatra" doesn't match "sinatra::ShowExceptions".
pub(super) fn file_stem_prefix_matches_identifier(stem: &str, identifier: &str) -> bool {
    let stem_lower = stem.to_ascii_lowercase();
    let ident_lower = identifier.to_ascii_lowercase();
    let bare_ident = ident_lower
        .rsplit_once("::")
        .map(|(_, name)| name)
        .unwrap_or(&ident_lower);
    common_prefix_len(&stem_lower, bare_ident) >= 6
}

pub(super) fn identifier_matches_parent_dir(identifier: &str, path: &str) -> bool {
    parent_dir_stems(path)
        .iter()
        .any(|stem| stem.eq_ignore_ascii_case(identifier))
}

pub(super) fn parent_dir_stems(path: &str) -> Vec<String> {
    let Some((dirs, _)) = path.rsplit_once('/') else {
        return Vec::new();
    };
    dirs.split('/')
        .rev()
        .take(3)
        .flat_map(identifier_stems)
        .collect()
}

pub(super) fn is_public_symbol(result: &SearchResult) -> bool {
    content_declares_public_symbol(&result.content)
}

pub(super) fn looks_like_impl_block(result: &SearchResult) -> bool {
    content_starts_with_impl(&result.content)
}

pub(super) fn shares_keyword_stem(left: &str, right: &str) -> bool {
    // Use minimum 4-char prefix overlap so short stems like "route" match
    // "routing" and "depend" matches "dependency". Longer words use longer
    // thresholds to avoid false positives.
    let shorter = left.len().min(right.len());
    let threshold = if shorter <= 5 { 4 } else { 5 };
    common_prefix_len(left, right) >= threshold
}

pub(super) fn common_prefix_len(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(l, r)| l == r)
        .count()
}

pub(super) fn symbol_keyword_bonus(symbol_name: &str, keywords: &[String]) -> f64 {
    let tokens = identifier_stems(symbol_name);

    if tokens.is_empty() {
        return 0.0;
    }

    if tokens
        .iter()
        .any(|token| keywords.iter().any(|keyword| keyword == token))
    {
        return 0.5;
    }

    if tokens.iter().any(|token| {
        keywords
            .iter()
            .any(|keyword| shares_keyword_stem(token, keyword))
    }) {
        return 0.32;
    }

    0.0
}

pub(super) fn identifier_stems(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .flat_map(split_camel_identifier)
        .map(|part| normalize_token(&part))
        .filter(|part| !part.is_empty() && !is_query_stopword(part))
        .collect()
}

pub(super) fn split_camel_identifier(value: &str) -> Vec<String> {
    if value.is_empty() {
        return Vec::new();
    }

    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = value.char_indices().collect();
    for idx in 1..chars.len() {
        let (_, prev) = chars[idx - 1];
        let (byte_idx, current) = chars[idx];
        let boundary = (prev.is_ascii_lowercase() && current.is_ascii_uppercase())
            || (prev.is_ascii_alphabetic() && current.is_ascii_digit())
            || (prev.is_ascii_digit() && current.is_ascii_alphabetic());
        if boundary {
            parts.push(value[start..byte_idx].to_ascii_lowercase());
            start = byte_idx;
        }
    }
    parts.push(value[start..].to_ascii_lowercase());
    parts
}

pub(super) fn is_definition_symbol(symbol_type: Option<SymbolType>) -> bool {
    matches!(
        symbol_type,
        Some(
            SymbolType::Class
                | SymbolType::Struct
                | SymbolType::Trait
                | SymbolType::Interface
                | SymbolType::Enum
                | SymbolType::Function
                | SymbolType::Method
                | SymbolType::Module
        )
    )
}

pub(super) fn is_compat_path(path: &str) -> bool {
    let tokens = tokenize_path(path);
    contains_token(
        &tokens,
        &[
            "compat",
            "compatibility",
            "legacy",
            "shim",
            "shims",
            "polyfill",
            "polyfills",
        ],
    )
}

pub(super) fn is_typescript_declaration(path: &str) -> bool {
    path.ends_with(".d.ts") || path.ends_with(".d.mts") || path.ends_with(".d.cts")
}

pub(super) fn is_reexport_barrel(result: &SearchResult) -> bool {
    let filename = file_name(&result.file_path).to_ascii_lowercase();
    if !matches!(
        filename.as_str(),
        "index.ts" | "index.tsx" | "index.js" | "index.jsx" | "mod.rs"
    ) {
        return false;
    }

    let non_empty: Vec<&str> = result
        .content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .collect();
    if non_empty.is_empty() || non_empty.len() > 24 {
        return false;
    }

    let reexports = non_empty
        .iter()
        .filter(|line| {
            line.starts_with("export ")
                || line.starts_with("pub use ")
                || line.starts_with("pub mod ")
                || line.starts_with("module.exports")
        })
        .count();
    reexports > 0 && reexports * 4 >= non_empty.len() * 3
}

pub(super) fn version_path_bonus(features: &QueryFeatures, path: &str) -> f64 {
    if features.requested_versions.is_empty() {
        return 0.0;
    }

    let tokens = tokenize_path(path);
    if tokens.iter().any(|token| {
        features
            .requested_versions
            .iter()
            .any(|version| version == token)
    }) {
        return 0.55;
    }

    if tokens.iter().any(|token| {
        token.len() >= 2
            && token.starts_with('v')
            && token[1..].chars().all(|ch| ch.is_ascii_digit())
    }) {
        return -0.34;
    }

    -0.08
}

pub(super) fn prefers_structural_chunks(features: &QueryFeatures) -> bool {
    features.query_type == QueryType::NaturalLanguage
        && features.exact_identifier.is_none()
        && features.query_word_count >= 4
        && !features.wants_config_paths
}

pub(super) fn structural_chunk_bias(result: &SearchResult) -> f64 {
    let lines = chunk_line_span(result);
    let mut bonus = 0.0;

    match result.symbol_type {
        Some(
            SymbolType::Struct | SymbolType::Class | SymbolType::Trait | SymbolType::Interface,
        ) => {
            bonus += 0.38;
        }
        Some(SymbolType::Enum | SymbolType::Module) => {
            bonus += 0.28;
        }
        Some(SymbolType::Block) if looks_like_impl_block(result) || lines >= 24 => {
            bonus += 0.24;
        }
        Some(SymbolType::Variable) => {
            bonus -= 0.45;
        }
        Some(SymbolType::Method | SymbolType::Function) if lines <= 8 => {
            bonus -= 0.32;
        }
        _ => {}
    }

    if lines <= 4 {
        bonus -= 0.2;
    } else if (12..=120).contains(&lines) {
        bonus += 0.12;
    }

    bonus
}

pub(super) fn chunk_line_span(result: &SearchResult) -> u32 {
    result.line_end.saturating_sub(result.line_start) + 1
}

/// Extract CamelCase/camelCase identifiers embedded in NL queries.
///
/// "How does StateManager handle transitions" → ["statemanager"]
/// "Where is the parseConfig function" → ["parseconfig"]
///
/// These are compound identifiers that contain mixed case transitions,
/// indicating a specific code symbol the user is asking about.
pub(super) fn extract_embedded_symbols(
    raw_tokens: &[&str],
    exact_identifier: Option<&str>,
) -> Vec<String> {
    let exact_lower = exact_identifier.map(|s| s.to_ascii_lowercase());
    raw_tokens
        .iter()
        .filter_map(|token| {
            let trimmed = trim_query_token(token);
            if trimmed.len() < 4 {
                return None;
            }
            // Must have a case transition (CamelCase or camelCase).
            let has_case_transition = trimmed
                .as_bytes()
                .windows(2)
                .any(|pair| pair[0].is_ascii_lowercase() && pair[1].is_ascii_uppercase());
            if !has_case_transition {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            // Skip if this is already the exact_identifier (already boosted separately).
            if exact_lower.as_deref() == Some(&lower) {
                return None;
            }
            Some(lower)
        })
        .collect()
}

/// Check if chunk content defines a symbol using language-agnostic keyword matching.
///
/// Looks for definition keywords (class, struct, def, function, etc.) followed
/// by a symbol name that matches query keywords. This is stronger than just
/// checking symbol_type metadata because it confirms the chunk is the actual
/// definition site, not just a reference.
pub(super) fn content_defines_query_keyword(content: &str, keywords: &[String]) -> bool {
    static DEFINITION_PREFIXES: &[&str] = &[
        "class ",
        "struct ",
        "enum ",
        "trait ",
        "interface ",
        "type ",
        "module ",
        "def ",
        "fn ",
        "func ",
        "function ",
        "fun ",
        "pub fn ",
        "pub struct ",
        "pub enum ",
        "pub trait ",
        "pub type ",
        "pub mod ",
        "export class ",
        "export function ",
        "export interface ",
        "export type ",
        "export enum ",
        "export default class ",
        "export default function ",
        "abstract class ",
        "data class ",
        "object ",
        "protocol ",
        "record ",
        "namespace ",
        "package ",
        "defmodule ",
    ];

    // Only consider keywords with 5+ chars to avoid false positives
    // from common short words like "file", "type", "list".
    let long_keywords: Vec<&String> = keywords.iter().filter(|k| k.len() >= 5).collect();
    if long_keywords.is_empty() {
        return false;
    }

    for line in content.lines().take(5) {
        let trimmed = line.trim();
        for prefix in DEFINITION_PREFIXES {
            let rest = trimmed.strip_prefix(*prefix);
            if let Some(rest) = rest {
                // Extract the symbol name after the keyword.
                let symbol: String = rest
                    .chars()
                    .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                    .collect();
                if symbol.len() >= 3 {
                    let sym_stems = identifier_stems(&symbol);
                    let matches_keyword = long_keywords.iter().any(|kw| {
                        sym_stems
                            .iter()
                            .any(|s| s == kw.as_str() || shares_keyword_stem(s, kw))
                    });
                    if matches_keyword {
                        return true;
                    }
                }
            }
        }
    }
    false
}
