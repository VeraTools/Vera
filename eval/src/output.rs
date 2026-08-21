//! Output formatting for evaluation results.
//!
//! Produces both machine-readable JSON and human-readable summary output.

use anyhow::Result;
use std::io::Write;
use std::path::Path;

use crate::types::EvalReport;

/// Write one report as an object or multiple reports as an array.
pub fn write_json_reports(reports: &[EvalReport], path: &Path) -> Result<()> {
    let json = reports_to_json(reports)?;
    std::fs::write(path, &json)?;
    Ok(())
}

/// Serialize the evaluation report to a JSON string.
pub fn report_to_json(report: &EvalReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

/// Serialize one report as an object or multiple reports as an array.
pub fn reports_to_json(reports: &[EvalReport]) -> Result<String> {
    match reports {
        [report] => report_to_json(report),
        _ => Ok(serde_json::to_string_pretty(reports)?),
    }
}

/// Print a human-readable summary of the evaluation report.
pub fn print_summary(report: &EvalReport, writer: &mut dyn Write) -> Result<()> {
    writeln!(
        writer,
        "╔══════════════════════════════════════════════════╗"
    )?;
    writeln!(
        writer,
        "║        Vera Evaluation Report                   ║"
    )?;
    writeln!(
        writer,
        "╚══════════════════════════════════════════════════╝"
    )?;
    writeln!(writer)?;
    writeln!(writer, "Tool:      {}", report.tool_name)?;
    writeln!(writer, "Timestamp: {}", report.timestamp)?;
    writeln!(writer, "Version:   {}", report.version_info.tool_version)?;
    writeln!(writer, "Tasks:     {}", report.per_task.len())?;
    writeln!(writer)?;

    // Overall aggregate metrics
    writeln!(
        writer,
        "── Overall Retrieval Metrics ──────────────────────"
    )?;
    let r = &report.aggregate.retrieval;
    writeln!(writer, "  Recall@1:  {:.4}", r.recall_at_1)?;
    writeln!(writer, "  Recall@5:  {:.4}", r.recall_at_5)?;
    writeln!(writer, "  Recall@10: {:.4}", r.recall_at_10)?;
    writeln!(writer, "  MRR:       {:.4}", r.mrr)?;
    writeln!(writer, "  nDCG:      {:.4}", r.ndcg)?;
    writeln!(writer)?;

    // Performance metrics
    writeln!(
        writer,
        "── Performance Metrics ────────────────────────────"
    )?;
    let p = &report.aggregate.performance;
    writeln!(writer, "  Latency p50:    {:.2} ms", p.latency_p50_ms)?;
    writeln!(writer, "  Latency p95:    {:.2} ms", p.latency_p95_ms)?;
    writeln!(writer, "  Index time:     {:.2} s", p.index_time_secs)?;
    writeln!(
        writer,
        "  Storage size:   {}",
        format_bytes(p.storage_size_bytes)
    )?;
    writeln!(writer, "  Token count:    {}", p.total_token_count)?;
    writeln!(writer)?;

    // Per-category breakdown
    writeln!(
        writer,
        "── Per-Category Breakdown ─────────────────────────"
    )?;
    writeln!(
        writer,
        "  {:<16} {:>8} {:>8} {:>8} {:>8} {:>8} {:>5}",
        "Category", "R@1", "R@5", "R@10", "MRR", "nDCG", "Tasks"
    )?;
    writeln!(writer, "  {}", "─".repeat(63))?;

    let mut categories: Vec<_> = report.per_category.iter().collect();
    categories.sort_by_key(|(k, _)| (*k).clone());

    for (cat, agg) in &categories {
        let r = &agg.retrieval;
        writeln!(
            writer,
            "  {:<16} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>5}",
            cat, r.recall_at_1, r.recall_at_5, r.recall_at_10, r.mrr, r.ndcg, agg.task_count
        )?;
    }
    writeln!(writer)?;

    // Per-task details
    writeln!(
        writer,
        "── Per-Task Results ───────────────────────────────"
    )?;
    writeln!(
        writer,
        "  {:<20} {:<16} {:>6} {:>6} {:>7} {:>6} {:>8}",
        "Task", "Category", "R@1", "R@10", "MRR", "nDCG", "Lat(ms)"
    )?;
    writeln!(writer, "  {}", "─".repeat(72))?;

    for eval in &report.per_task {
        let r = &eval.retrieval_metrics;
        writeln!(
            writer,
            "  {:<20} {:<16} {:>6.2} {:>6.2} {:>7.4} {:>6.2} {:>8.2}",
            truncate(&eval.task_id, 20),
            eval.category.to_string(),
            r.recall_at_1,
            r.recall_at_10,
            r.mrr,
            r.ndcg,
            eval.latency_ms
        )?;
    }
    writeln!(writer)?;

    // Version info
    writeln!(
        writer,
        "── Reproducibility Info ───────────────────────────"
    )?;
    writeln!(
        writer,
        "  Corpus version: {}",
        report.version_info.corpus_version
    )?;
    if !report.version_info.repo_shas.is_empty() {
        writeln!(writer, "  Repo SHAs:")?;
        let mut shas: Vec<_> = report.version_info.repo_shas.iter().collect();
        shas.sort_by_key(|(k, _)| (*k).clone());
        for (name, sha) in &shas {
            writeln!(writer, "    {:<15} {}", name, &sha[..sha.len().min(12)])?;
        }
    }
    if !report.version_info.config.is_empty() {
        writeln!(writer, "  Config:")?;
        let mut config: Vec<_> = report.version_info.config.iter().collect();
        config.sort_by_key(|(k, _)| (*k).clone());
        for (key, val) in &config {
            writeln!(writer, "    {}: {}", key, val)?;
        }
    }
    writeln!(writer)?;

    Ok(())
}

/// Format bytes into a human-readable string.
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Truncate a string to a maximum length, adding "…" if truncated.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AggregateMetrics, PerformanceMetrics, RetrievalMetrics, TaskCategory, TaskEvaluation,
        VersionInfo,
    };
    use std::collections::HashMap;

    fn sample_report() -> EvalReport {
        let retrieval = RetrievalMetrics {
            recall_at_1: 0.8,
            recall_at_5: 0.9,
            recall_at_10: 1.0,
            mrr: 0.85,
            ndcg: 0.9,
        };
        let performance = PerformanceMetrics {
            latency_p50_ms: 10.0,
            latency_p95_ms: 25.0,
            index_time_secs: 5.0,
            storage_size_bytes: 2 * 1024 * 1024,
            total_token_count: 500,
        };

        EvalReport {
            tool_name: "test-tool".to_string(),
            timestamp: "2026-03-20T00:00:00Z".to_string(),
            version_info: VersionInfo {
                tool_version: "0.1.0".to_string(),
                corpus_version: 1,
                repo_shas: HashMap::from([("ripgrep".to_string(), "abc123".to_string())]),
                config: HashMap::new(),
                lane: None,
                task_set: None,
                vera_git_sha: None,
                command: Vec::new(),
                environment: Default::default(),
            },
            per_task: vec![TaskEvaluation {
                task_id: "test-001".to_string(),
                category: TaskCategory::SymbolLookup,
                retrieval_metrics: retrieval.clone(),
                latency_ms: 10.0,
                result_count: 5,
                results: Vec::new(),
            }],
            per_category: HashMap::from([(
                "symbol_lookup".to_string(),
                AggregateMetrics {
                    retrieval: retrieval.clone(),
                    performance: performance.clone(),
                    task_count: 1,
                },
            )]),
            aggregate: AggregateMetrics {
                retrieval,
                performance,
                task_count: 1,
            },
        }
    }

    #[test]
    fn test_json_output_parseable() {
        let report = sample_report();
        let json = report_to_json(&report).unwrap();

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["tool_name"], "test-tool");
        assert!(
            parsed["aggregate"]["retrieval"]["recall_at_1"]
                .as_f64()
                .is_some()
        );
    }

    #[test]
    fn test_json_has_all_metric_families() {
        let report = sample_report();
        let json = report_to_json(&report).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Retrieval metrics
        let retrieval = &parsed["aggregate"]["retrieval"];
        assert!(retrieval["recall_at_1"].is_number());
        assert!(retrieval["recall_at_5"].is_number());
        assert!(retrieval["recall_at_10"].is_number());
        assert!(retrieval["mrr"].is_number());
        assert!(retrieval["ndcg"].is_number());

        // Performance metrics
        let perf = &parsed["aggregate"]["performance"];
        assert!(perf["latency_p50_ms"].is_number());
        assert!(perf["latency_p95_ms"].is_number());
        assert!(perf["index_time_secs"].is_number());
        assert!(perf["storage_size_bytes"].is_number());
        assert!(perf["total_token_count"].is_number());
    }

    #[test]
    fn test_human_readable_output() {
        let report = sample_report();
        let mut output = Vec::new();
        print_summary(&report, &mut output).unwrap();
        let text = String::from_utf8(output).unwrap();

        assert!(text.contains("Vera Evaluation Report"));
        assert!(text.contains("test-tool"));
        assert!(text.contains("Recall@1"));
        assert!(text.contains("Recall@5"));
        assert!(text.contains("Recall@10"));
        assert!(text.contains("MRR"));
        assert!(text.contains("nDCG"));
        assert!(text.contains("Latency p50"));
        assert!(text.contains("Latency p95"));
        assert!(text.contains("Index time"));
        assert!(text.contains("Storage size"));
        assert!(text.contains("Token count"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.0 MB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GB");
    }

    #[test]
    fn test_write_json_reports() {
        let report = sample_report();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        write_json_reports(std::slice::from_ref(&report), &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let _: EvalReport = serde_json::from_str(&content).unwrap();
    }

    #[test]
    fn test_multiple_reports_are_written_as_an_array() {
        let report = sample_report();
        let json = reports_to_json(&[report.clone(), report]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }
}
