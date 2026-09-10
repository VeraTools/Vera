//! `vera stats` — Show index statistics.

/// Run the `vera stats` command.
pub fn run(json_output: bool) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("failed to get current directory: {e}"))?;
    // Match the read commands: an ancestor `.vera/` index covers this
    // subdirectory too.
    let repo_root = crate::helpers::resolve_index_root(&cwd).unwrap_or_else(|| cwd.clone());

    let stats = vera_core::stats::collect_stats(&repo_root)?;

    if json_output {
        let json = serde_json::to_string_pretty(&stats)
            .map_err(|e| anyhow::anyhow!("failed to serialize stats: {e}"))?;
        println!("{json}");
    } else {
        print_human_stats(&stats);
    }

    Ok(())
}

/// Print human-readable index statistics.
fn print_human_stats(stats: &vera_core::stats::IndexStats) {
    println!("Index Statistics");
    println!();
    println!("  Files indexed:   {}", stats.file_count);
    println!("  Total chunks:    {}", stats.chunk_count);
    println!("  Index size:      {}", stats.index_size_human);

    println!();
    println!("  Index Health:");
    println!(
        "    Tree-sitter errors: {}",
        stats.index_health.files_with_tree_sitter_errors
    );
    println!(
        "    Tier 0 fallback:    {}",
        stats.index_health.files_using_tier0_fallback
    );
    println!(
        "    Parse failures:     {}",
        stats.index_health.files_with_parse_failures
    );
    println!();
    if !stats.languages.is_empty() {
        println!("  Language Breakdown:");
        for lang in &stats.languages {
            println!(
                "    {:<15} {:>5} chunks ({:.1}%)",
                lang.language, lang.chunk_count, lang.percentage
            );
        }
    } else {
        println!("  No languages indexed.");
    }

    if !stats.index_health.by_language.is_empty() {
        println!();
        println!("  Health by Language:");
        for lang in &stats.index_health.by_language {
            println!(
                "    {:<15} indexed {:>4}  errors {:>3}  tier0 {:>3}  parse {:>3}",
                lang.language,
                lang.files_indexed,
                lang.files_with_tree_sitter_errors,
                lang.files_using_tier0_fallback,
                lang.files_with_parse_failures,
            );
        }
    }
}
