//! `vera grep <pattern>` — Regex search over indexed files.

use std::io::Write;
use std::time::Instant;

use crate::helpers::{output_results, prepare_indexed_search};
use crate::state;

/// Run the `vera grep <pattern>` command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    pattern: &str,
    limit: Option<usize>,
    case_insensitive: bool,
    context_lines: usize,
    filters: &vera_core::types::SearchFilters,
    json_output: bool,
    raw: bool,
    timing: bool,
    git_scope: Option<vera_core::git_scope::GitScope>,
    compact: bool,
) -> anyhow::Result<()> {
    let config = state::load_runtime_config()?;
    let (index_dir, filters) =
        prepare_indexed_search(&config.indexing, filters, git_scope.as_ref())?;

    let result_limit = limit.unwrap_or(20);
    let started_at = Instant::now();
    let results = vera_core::retrieval::search_regex(
        &index_dir,
        pattern,
        result_limit,
        case_insensitive,
        context_lines,
        &filters,
    )?;

    output_results(
        &results,
        json_output,
        raw,
        compact,
        config.retrieval.max_output_chars,
    );

    if results.is_empty() {
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        if let Some(hint) = alternation_hint(pattern) {
            let _ = writeln!(err, "{hint}");
        }
        if let Some(hint) = crate::helpers::path_filter_hint(&index_dir, &filters) {
            let _ = writeln!(err, "{hint}");
        }
    }

    if timing {
        let elapsed = started_at.elapsed();
        let stderr = std::io::stderr();
        let mut err = stderr.lock();
        let _ = writeln!(err, "[timing] total: {}ms", elapsed.as_millis());
    }

    Ok(())
}

fn alternation_hint(pattern: &str) -> Option<String> {
    if !pattern.contains(r"\|") {
        return None;
    }

    let suggested = pattern.replace(r"\|", "|");
    if suggested == pattern {
        return None;
    }

    Some(format!(
        "hint: `vera grep` uses Rust regex syntax, so `\\|` matches a literal pipe. If you meant alternation, try `{suggested}`."
    ))
}

#[cfg(test)]
mod tests {
    use super::alternation_hint;

    #[test]
    fn alternation_hint_suggests_plain_pipe() {
        let hint =
            alternation_hint(r"persist_kimi_auth_record\|persist_factory_auth_record").unwrap();
        assert!(hint.contains("persist_kimi_auth_record|persist_factory_auth_record"));
    }

    #[test]
    fn alternation_hint_ignores_normal_patterns() {
        assert!(alternation_hint("TODO|FIXME").is_none());
    }
}
