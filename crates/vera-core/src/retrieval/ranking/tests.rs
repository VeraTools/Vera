use super::score::*;
use super::*;
use crate::config::VeraConfig;
use crate::types::SymbolType;

fn make_result(
    file_path: &str,
    symbol_name: Option<&str>,
    symbol_type: Option<SymbolType>,
    content: &str,
) -> SearchResult {
    SearchResult {
        file_path: file_path.to_string(),
        line_start: 1,
        line_end: 20,
        content: content.to_string(),
        language: Language::Rust,
        score: 1.0,
        symbol_name: symbol_name.map(ToString::to_string),
        symbol_type,
        part_index: None,
    }
}

#[test]
fn root_config_file_beats_nested_match() {
    let results = vec![
        make_result(
            "fuzz/Cargo.toml",
            Some("Cargo.toml"),
            Some(SymbolType::Block),
            "[package]\nname = \"fuzz\"",
        ),
        make_result(
            "Cargo.toml",
            Some("Cargo.toml"),
            Some(SymbolType::Block),
            "[workspace]\nmembers = [\"crates/vera-core\"]",
        ),
    ];

    let ranked = apply_query_ranking(
        "Cargo.toml workspace configuration",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "Cargo.toml");
}

#[test]
fn test_paths_are_demoted_for_non_test_queries() {
    let results = vec![
        make_result(
            "tests/validation.test.ts",
            Some("validation"),
            Some(SymbolType::Function),
            "export function validation() {}",
        ),
        make_result(
            "src/validation.ts",
            Some("validateRequest"),
            Some(SymbolType::Function),
            "export function validateRequest() {}",
        ),
    ];

    let ranked = apply_query_ranking(
        "request validation and schema enforcement",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "src/validation.ts");
}

#[test]
fn requested_symbol_type_gets_priority() {
    let results = vec![
        make_result(
            "src/blueprint_methods.py",
            Some("register"),
            Some(SymbolType::Method),
            "def register(self): pass",
        ),
        make_result(
            "src/blueprint.py",
            Some("Blueprint"),
            Some(SymbolType::Class),
            "class Blueprint:\n    pass",
        ),
    ];

    let ranked = apply_query_ranking("Blueprint class definition", results, RankingStage::Initial);

    assert_eq!(ranked[0].symbol_type, Some(SymbolType::Class));
}

#[test]
fn case_exact_identifier_beats_lowercase_method() {
    let results = vec![
        make_result(
            "src/server.rs",
            Some("run"),
            Some(SymbolType::Method),
            "fn run(&self) {}",
        ),
        make_result(
            "src/run/mod.rs",
            Some("Run"),
            Some(SymbolType::Struct),
            "pub struct Run {}",
        ),
    ];

    let ranked = apply_query_ranking("Run struct definition", results, RankingStage::Initial);

    assert_eq!(ranked[0].symbol_name.as_deref(), Some("Run"));
}

#[test]
fn public_class_definition_beats_internal_variant() {
    let results = vec![
        make_result(
            "src/flask/sansio/blueprints.py",
            Some("Blueprint"),
            Some(SymbolType::Class),
            "class Blueprint:\n    pass",
        ),
        make_result(
            "src/flask/blueprints.py",
            Some("Blueprint"),
            Some(SymbolType::Class),
            "class Blueprint:\n    pass",
        ),
    ];

    let ranked = apply_query_ranking("Blueprint class definition", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/flask/blueprints.py");
}

#[test]
fn natural_language_queries_promote_file_diversity() {
    // When multiple files have similar relevance, diversity should
    // interleave them rather than clustering same-file results.
    let results = vec![
        make_result(
            "src/router.ts",
            Some("register_routes"),
            Some(SymbolType::Function),
            "export function register_routes() {}",
        ),
        make_result(
            "src/router.ts",
            Some("add_route"),
            Some(SymbolType::Function),
            "export function add_route() {}",
        ),
        make_result(
            "src/blueprint.ts",
            Some("create_blueprint"),
            Some(SymbolType::Function),
            "export function create_blueprint() {}",
        ),
    ];

    let ranked = apply_query_ranking(
        "Blueprint registration and route mounting",
        results,
        RankingStage::Initial,
    );

    // blueprint.ts matches the query keyword "blueprint", so diversity
    // should interleave it between the two router.ts chunks.
    assert_eq!(ranked[0].file_path, "src/router.ts");
    assert_eq!(ranked[1].file_path, "src/blueprint.ts");
}

#[test]
fn explicit_path_fragment_beats_root_config_bias() {
    let results = vec![
        make_result(
            "Cargo.toml",
            Some("Cargo.toml"),
            Some(SymbolType::Block),
            "[workspace]\nmembers = [\"crates/vera-core\"]",
        ),
        make_result(
            "fuzz/Cargo.toml",
            Some("Cargo.toml"),
            Some(SymbolType::Block),
            "[package]\nname = \"fuzz\"",
        ),
    ];

    let ranked = apply_query_ranking(
        "fuzz/Cargo.toml package manifest",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "fuzz/Cargo.toml");
}

#[test]
fn testing_module_is_treated_like_test_noise() {
    let results = vec![
        make_result(
            "src/flask/testing.py",
            Some("session_transaction"),
            Some(SymbolType::Method),
            "def session_transaction(self): pass",
        ),
        make_result(
            "src/flask/sessions.py",
            Some("save_session"),
            Some(SymbolType::Method),
            "def save_session(self): pass",
        ),
    ];

    let ranked = apply_query_ranking(
        "session management and cookie handling",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "src/flask/sessions.py");
}

#[test]
fn broad_code_queries_prefer_source_over_docs() {
    let results = vec![
        make_result(
            "docs/Reference/Validation-and-Serialization.md",
            None,
            None,
            "Validation and serialization documentation.",
        ),
        make_result(
            "lib/validation.js",
            Some("validate"),
            Some(SymbolType::Function),
            "function validateRequestSchema () {}",
        ),
    ];

    let ranked = apply_query_ranking(
        "request validation and schema enforcement",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "lib/validation.js");
}

#[test]
fn archived_docs_are_demoted_for_exact_queries() {
    let results = vec![
        make_result(
            "archive/docs/hotkeys.md",
            None,
            None,
            "keybind guide and notes",
        ),
        make_result(
            "src/mod_content/hotkeys.ts",
            Some("registerHotkeys"),
            Some(SymbolType::Function),
            "export function registerHotkeys() {}",
        ),
    ];

    let ranked = apply_query_ranking("hotkeys keybind", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/mod_content/hotkeys.ts");
}

#[test]
fn runtime_queries_can_prefer_runtime_extracts() {
    let results = vec![
        make_result(
            "src/mod_loader.ts",
            Some("loadMod"),
            Some(SymbolType::Function),
            "export function loadMod() {}",
        ),
        make_result(
            "/tmp/installed-game-runtime/Game.pretty.js",
            Some("loadMod"),
            Some(SymbolType::Function),
            "function loadMod() {}",
        ),
    ];

    let ranked = apply_query_ranking("runtime mod loader extract", results, RankingStage::Initial);

    assert_eq!(
        ranked[0].file_path,
        "/tmp/installed-game-runtime/Game.pretty.js"
    );
}

#[test]
fn content_coverage_breaks_filename_stem_ties() {
    // Both stems match one query keyword ("request" / "validation"), so the
    // pool-relative stem boost ties. Coverage breaks the tie: the helper's
    // require line mentions "validation" and "schema" while the stub covers
    // nothing. (The retired exact-stem tier would have forced validation.js
    // first, but ablation showed the coverage-aware ordering ranks better on
    // both benchmark sets.)
    let results = vec![
        make_result(
            "lib/handle-request.js",
            None,
            Some(SymbolType::Variable),
            "const validateSchema = require('./validation')",
        ),
        make_result(
            "lib/validation.js",
            None,
            Some(SymbolType::Variable),
            "function validate () {}",
        ),
    ];

    let ranked = apply_query_ranking(
        "request validation and schema enforcement",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "lib/handle-request.js");
}

#[test]
fn stem_matched_file_with_content_coverage_wins() {
    // When the stem-matched file also covers the query's concepts in its
    // content and retrieval ranked it first, it keeps the lead: stem
    // agreement plus coverage beats either signal alone.
    let results = vec![
        make_result(
            "lib/validation.js",
            None,
            Some(SymbolType::Variable),
            "function validate (request, schema) { return enforce(schema); }",
        ),
        make_result(
            "lib/handle-request.js",
            None,
            Some(SymbolType::Variable),
            "const helper = require('./helper')",
        ),
    ];

    let ranked = apply_query_ranking(
        "request validation and schema enforcement",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "lib/validation.js");
}

#[test]
fn fuzzy_filename_stem_match_beats_unrelated_chunk() {
    let results = vec![
        make_result(
            "src/flask/sansio/blueprints.py",
            Some("BlueprintSetupState"),
            Some(SymbolType::Class),
            "class BlueprintSetupState:\n    pass",
        ),
        make_result(
            "src/flask/templating.py",
            Some("render_template"),
            Some(SymbolType::Function),
            "def render_template(template_name_or_list, **context):\n    return _render(...)",
        ),
    ];

    let ranked = apply_query_ranking(
        "template rendering pipeline",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "src/flask/templating.py");
}

#[test]
fn file_stem_prefix_match_ignores_namespace_prefixes() {
    assert!(file_stem_prefix_matches_identifier("format", "formatter"));
    assert!(!file_stem_prefix_matches_identifier(
        "sinatra",
        "sinatra::showexceptions"
    ));
}

#[test]
fn explicit_symbol_type_penalizes_mismatched_results() {
    let results = vec![
        make_result(
            "src/run.rs",
            Some("run"),
            Some(SymbolType::Function),
            "pub fn run() {}",
        ),
        make_result(
            "src/run/mod.rs",
            Some("Run"),
            Some(SymbolType::Struct),
            "pub struct Run {}",
        ),
    ];

    let ranked = apply_query_ranking("Run struct definition", results, RankingStage::Initial);

    assert_eq!(ranked[0].symbol_type, Some(SymbolType::Struct));
}

#[test]
fn broad_intent_queries_prefer_structural_chunks() {
    let results = vec![
            SearchResult {
                file_path: "crates/ignore/src/types.rs".to_string(),
                line_start: 132,
                line_end: 137,
                content:
                    "pub fn file_type_def(&self) -> Option<&FileTypeDef> {\n    match self {\n        _ => None,\n    }\n}"
                        .to_string(),
                language: Language::Rust,
                score: 1.0,
                symbol_name: Some("file_type_def".to_string()),
                symbol_type: Some(SymbolType::Method),
                part_index: None,
            },
            SearchResult {
                file_path: "crates/ignore/src/types.rs".to_string(),
                line_start: 165,
                line_end: 181,
                content: "pub struct Types {\n    defs: Vec<FileTypeDef>,\n    selections: Vec<String>,\n    set: GlobSet,\n}"
                    .to_string(),
                language: Language::Rust,
                score: 1.0,
                symbol_name: Some("Types".to_string()),
                symbol_type: Some(SymbolType::Struct),
                part_index: None,
            },
        ];

    let ranked = apply_query_ranking(
        "file type detection and filtering",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].symbol_name.as_deref(), Some("Types"));
}

#[test]
fn file_coherence_boost_promotes_repeated_relevant_file() {
    let results = vec![
        make_result(
            "src/misc.rs",
            Some("misc"),
            Some(SymbolType::Function),
            "pub fn misc() {}",
        ),
        make_result(
            "src/auth/session.rs",
            Some("renewSession"),
            Some(SymbolType::Function),
            "pub fn renew_session() {}",
        ),
        make_result(
            "src/auth/session.rs",
            Some("validateSession"),
            Some(SymbolType::Function),
            "pub fn validate_session() {}",
        ),
    ];

    let ranked = apply_query_ranking(
        "session renewal and validation flow",
        results,
        RankingStage::Initial,
    );

    assert_eq!(ranked[0].file_path, "src/auth/session.rs");
}

#[test]
fn parent_directory_stem_match_beats_flat_unrelated_file() {
    let results = vec![
        make_result(
            "src/router.rs",
            Some("route"),
            Some(SymbolType::Function),
            "pub fn route() {}",
        ),
        make_result(
            "src/auth/middleware.rs",
            Some("middleware"),
            Some(SymbolType::Function),
            "pub fn middleware() {}",
        ),
    ];

    let ranked = apply_query_ranking("auth middleware routing", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/auth/middleware.rs");
}

#[test]
fn compatibility_paths_are_demoted_unless_requested() {
    let results = vec![
        make_result(
            "src/compat/session.rs",
            Some("session"),
            Some(SymbolType::Function),
            "pub fn session() {}",
        ),
        make_result(
            "src/session.rs",
            Some("session"),
            Some(SymbolType::Function),
            "pub fn session() {}",
        ),
    ];

    let ranked = apply_query_ranking("session handling", results, RankingStage::Initial);
    assert_eq!(ranked[0].file_path, "src/session.rs");

    let ranked = apply_query_ranking("legacy compat session", ranked, RankingStage::Initial);
    assert_eq!(ranked[0].file_path, "src/compat/session.rs");
}

#[test]
fn version_intent_prefers_matching_path() {
    let results = vec![
        make_result(
            "src/v3/router.rs",
            Some("router"),
            Some(SymbolType::Function),
            "pub fn router() {}",
        ),
        make_result(
            "src/v4/router.rs",
            Some("router"),
            Some(SymbolType::Function),
            "pub fn router() {}",
        ),
    ];

    let ranked = apply_query_ranking("v4 router", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/v4/router.rs");
}

#[test]
fn declaration_files_and_reexport_barrels_are_demoted() {
    let results = vec![
        make_result(
            "src/index.ts",
            Some("index"),
            Some(SymbolType::Module),
            "export { Session } from './session'\nexport { Auth } from './auth'",
        ),
        make_result(
            "src/session.d.ts",
            Some("Session"),
            Some(SymbolType::Interface),
            "export interface Session {}",
        ),
        make_result(
            "src/session.ts",
            Some("Session"),
            Some(SymbolType::Class),
            "export class Session {}",
        ),
    ];

    let ranked = apply_query_ranking("session implementation", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/session.ts");
}

#[test]
fn definition_queries_boost_symbol_definitions() {
    let results = vec![
        make_result(
            "src/parser.rs",
            Some("PARSER"),
            Some(SymbolType::Variable),
            "static PARSER: Parser = Parser::new();",
        ),
        make_result(
            "src/parser.rs",
            Some("Parser"),
            Some(SymbolType::Struct),
            "pub struct Parser {}",
        ),
    ];

    let ranked = apply_query_ranking("Parser definition", results, RankingStage::Initial);

    assert_eq!(ranked[0].symbol_type, Some(SymbolType::Struct));
}

#[test]
fn fixture_definition_in_tests_loses_content_symbol_boost() {
    // Both chunks define the queried symbol in content, but the test chunk is
    // a fixture, not the definition site: the content-symbol boost must not
    // apply to it. Without the role gate the +3.0 pool-relative boost would
    // outweigh the test-path penalty and rank the fixture first.
    let results = vec![
        make_result(
            "tests/test_registry.py",
            None,
            None,
            "class TaskRegistry:\n    def register(self, task): ...",
        ),
        make_result(
            "celery/app/registry.py",
            None,
            None,
            "class TaskRegistry:\n    def register(self, task): ...",
        ),
    ];

    let ranked = apply_query_ranking("TaskRegistry", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "celery/app/registry.py");
}

#[test]
fn testing_module_file_keeps_content_symbol_boost() {
    // click ships the CliRunner definition in src/click/testing.py: a
    // first-class module whose filename merely looks test-ish. The gate keys
    // off directory components and test-file naming conventions, so this
    // chunk keeps its definition boost despite the filename.
    let results = vec![
        make_result(
            "src/click/core.py",
            None,
            None,
            "class Command:\n    def invoke(self): ...",
        ),
        make_result(
            "src/click/testing.py",
            None,
            None,
            "class CliRunner:\n    def invoke(self, cli): ...",
        ),
    ];

    let ranked = apply_query_ranking("CliRunner", results, RankingStage::Initial);

    assert_eq!(ranked[0].file_path, "src/click/testing.py");
}

#[test]
fn path_weighted_query_requires_path_shaped_token() {
    // Slash prose stays semantic.
    assert!(!is_path_weighted_query("read/write request handling"));
    assert!(!is_path_weighted_query("client/server architecture"));
    assert!(!is_path_weighted_query("and/or logic in the parser"));

    // Real paths stay path-weighted.
    assert!(is_path_weighted_query("src/main.rs"));
    assert!(is_path_weighted_query("crates/vera-core/src"));
    assert!(is_path_weighted_query("how does src/main.rs work"));
    assert!(is_path_weighted_query("what lives in ./crates today"));
    assert!(is_path_weighted_query(r"c:\src\main.rs details"));

    // Config-filename mentions keep the existing substring behavior.
    assert!(is_path_weighted_query("config.toml loading"));
    assert!(is_path_weighted_query("dockerfile setup"));
}

// ——— Issue #196: toggleable ranking signals ———
#[test]
fn filename_stem_boost_is_toggleable() {
    // Two files with identical neutral content and symbols so only the
    // filename stem distinguishes them. The second file's path matches the
    // query keyword "rendering"; with the boost enabled it must outrank the
    // base leader, with it disabled the input order must hold.
    let neutral = "pub fn helper() {}";
    let results = vec![
        make_result(
            "src/auth/middleware.rs",
            Some("helper"),
            Some(SymbolType::Function),
            neutral,
        ),
        make_result(
            "src/rendering/engine.rs",
            Some("helper"),
            Some(SymbolType::Function),
            neutral,
        ),
    ];

    let ranked_enabled = apply_query_ranking_with_filters_and_config(
        "rendering engine pipeline",
        results.clone(),
        RankingStage::Initial,
        &SearchFilters::default(),
        &VeraConfig::default(),
    );
    assert_eq!(
        ranked_enabled[0].file_path, "src/rendering/engine.rs",
        "filename-stem boost should promote rendering/engine.rs when enabled"
    );

    let mut disabled = VeraConfig::default();
    disabled.retrieval.ranking_filename_stem_boost = false;
    let ranked_disabled = apply_query_ranking_with_filters_and_config(
        "rendering engine pipeline",
        results,
        RankingStage::Initial,
        &SearchFilters::default(),
        &disabled,
    );
    assert_eq!(
        ranked_disabled[0].file_path, "src/auth/middleware.rs",
        "with filename-stem boost disabled, stem match must not overturn base rank"
    );
}

#[test]
fn definition_boost_is_toggleable() {
    // Two source files, both non-test, neither blocked. The second file's
    // content defines the queried symbol `CliRunner` while the first does
    // not. With the definition boost enabled the second file must win;
    // disabled, base rank (first) must hold.
    let results = vec![
        make_result(
            "src/click/core.py",
            Some("Command"),
            Some(SymbolType::Class),
            "class Command:\n    def invoke(self): ...",
        ),
        make_result(
            "src/click/testing.py",
            None,
            None,
            "class CliRunner:\n    def invoke(self, cli): ...",
        ),
    ];

    let enabled = apply_query_ranking_with_filters_and_config(
        "CliRunner",
        results.clone(),
        RankingStage::Initial,
        &SearchFilters::default(),
        &VeraConfig::default(),
    );
    assert_eq!(
        enabled[0].file_path, "src/click/testing.py",
        "definition boost should promote the CliRunner definition site when enabled"
    );

    let mut disabled = VeraConfig::default();
    disabled.retrieval.ranking_definition_boost = false;
    let disabled_ranked = apply_query_ranking_with_filters_and_config(
        "CliRunner",
        results,
        RankingStage::Initial,
        &SearchFilters::default(),
        &disabled,
    );
    assert_eq!(
        disabled_ranked[0].file_path, "src/click/core.py",
        "with definition boost disabled, content definition must not overturn base rank"
    );
}

#[test]
fn file_coherence_boost_survives_definition_flag_toggle() {
    // Coherence boost is independent of definition boost: repeated-file
    // promotion must still work when the definition flag is off.
    let results = vec![
        make_result(
            "src/misc.rs",
            Some("misc"),
            Some(SymbolType::Function),
            "pub fn misc() {}",
        ),
        make_result(
            "src/auth/session.rs",
            Some("renewSession"),
            Some(SymbolType::Function),
            "pub fn renew_session() {}",
        ),
        make_result(
            "src/auth/session.rs",
            Some("validateSession"),
            Some(SymbolType::Function),
            "pub fn validate_session() {}",
        ),
    ];

    let mut disabled = VeraConfig::default();
    disabled.retrieval.ranking_definition_boost = false;
    let ranked = apply_query_ranking_with_filters_and_config(
        "session renewal and validation flow",
        results,
        RankingStage::Initial,
        &SearchFilters::default(),
        &disabled,
    );
    assert_eq!(
        ranked[0].file_path, "src/auth/session.rs",
        "coherence boost must still promote repeated file even with definition boost off"
    );
}
