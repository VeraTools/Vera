//! Integration tests for the parsing pipeline.
//!
//! Tests the full `parse_and_chunk` flow across all supported languages,
//! Tier 0 fallback, large symbol splitting, and edge cases.

use crate::config::IndexingConfig;
use crate::parsing::test_support::{default_config, find_chunk, parse};
use crate::parsing::{parse_and_chunk, parse_file_with_diagnostics};
use crate::types::{Language, SymbolType};

// =========================================================
// Rust tests
// =========================================================

#[test]
fn rust_functions_produce_chunks() {
    let source = r#"fn hello() {
    println!("hello");
}

fn world() -> i32 {
    42
}"#;
    let chunks = parse(source, "main.rs", Language::Rust);
    let funcs: Vec<_> = chunks
        .iter()
        .filter(|c| c.symbol_type == Some(SymbolType::Function))
        .collect();
    assert_eq!(funcs.len(), 2, "expected 2 function chunks");
    assert_eq!(funcs[0].symbol_name, Some("hello".to_string()));
    assert_eq!(funcs[1].symbol_name, Some("world".to_string()));
}

#[test]
fn rust_struct_and_impl() {
    let source = r#"struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}"#;
    let chunks = parse(source, "point.rs", Language::Rust);
    assert!(chunks.len() >= 2, "expected struct + method(s)");

    let struc = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Struct));
    assert!(struc.is_some(), "should have struct chunk");
    assert_eq!(struc.unwrap().symbol_name, Some("Point".to_string()));

    let method = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Method));
    assert!(method.is_some(), "should have method chunk");
    assert_eq!(method.unwrap().symbol_name, Some("new".to_string()));

    let impl_block = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Block) && c.content.contains("impl Point"));
    assert!(impl_block.is_some(), "should keep impl block chunk");
}

#[test]
fn rust_enum_and_trait() {
    let source = r#"enum Color {
    Red,
    Green,
    Blue,
}

trait Paintable {
    fn paint(&self, color: Color);
}"#;
    let chunks = parse(source, "paint.rs", Language::Rust);
    let enm = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Enum));
    assert!(enm.is_some(), "should have enum chunk");
    let trt = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Trait));
    assert!(trt.is_some(), "should have trait chunk");
}

#[test]
fn rust_type_alias() {
    let source = "type Result<T> = std::result::Result<T, MyError>;\n";
    let chunks = parse(source, "lib.rs", Language::Rust);
    let ta = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::TypeAlias));
    assert!(ta.is_some(), "should have type alias chunk");
}

// =========================================================
// Python tests
// =========================================================

#[test]
fn python_function_and_class_methods() {
    let source = r#"def greet(name):
    return f"Hello, {name}!"

class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id):
        return self.db.get(user_id)
"#;
    let chunks = parse(source, "service.py", Language::Python);
    // Now: function + __init__ method + get_user method (+ possible gap chunks)
    assert!(chunks.len() >= 3, "expected function + 2 class methods");

    let func = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Function));
    assert!(func.is_some());
    assert_eq!(func.unwrap().symbol_name, Some("greet".to_string()));

    // Class methods are now extracted individually.
    let init = chunks
        .iter()
        .find(|c| c.symbol_name == Some("__init__".to_string()));
    assert!(init.is_some(), "should find method __init__");
    assert_eq!(init.unwrap().symbol_type, Some(SymbolType::Method));

    let get_user = chunks
        .iter()
        .find(|c| c.symbol_name == Some("get_user".to_string()));
    assert!(get_user.is_some(), "should find method get_user");
    assert_eq!(get_user.unwrap().symbol_type, Some(SymbolType::Method));

    let class_chunk = chunks
        .iter()
        .find(|c| c.symbol_name == Some("UserService".to_string()));
    assert!(class_chunk.is_some(), "should retain the class chunk");
    assert_eq!(class_chunk.unwrap().symbol_type, Some(SymbolType::Class));
}

#[test]
fn python_decorated_function() {
    let source = r#"import functools

@functools.cache
def expensive_compute(n):
    return sum(range(n))
"#;
    let chunks = parse(source, "compute.py", Language::Python);
    let func = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Function));
    assert!(func.is_some(), "should extract decorated function");
}

// =========================================================
// TypeScript tests
// =========================================================

#[test]
fn typescript_function_interface_class() {
    let source = r#"function add(a: number, b: number): number {
    return a + b;
}

interface Config {
    host: string;
    port: number;
}

class Server {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    start(): void {
        console.log("starting");
    }
}
"#;
    let chunks = parse(source, "server.ts", Language::TypeScript);
    assert!(
        chunks.len() >= 5,
        "expected function + interface + class + methods"
    );

    let func = chunks
        .iter()
        .find(|c| c.symbol_name == Some("add".to_string()));
    assert!(func.is_some());
    assert_eq!(func.unwrap().symbol_type, Some(SymbolType::Function));

    let iface = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Config".to_string()));
    assert!(iface.is_some());
    assert_eq!(iface.unwrap().symbol_type, Some(SymbolType::Interface));

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Server".to_string()));
    assert!(cls.is_some());
    assert_eq!(cls.unwrap().symbol_type, Some(SymbolType::Class));

    let constructor = chunks
        .iter()
        .find(|c| c.symbol_name == Some("constructor".to_string()));
    assert!(constructor.is_some());
    assert_eq!(constructor.unwrap().symbol_type, Some(SymbolType::Method));

    let method = chunks
        .iter()
        .find(|c| c.symbol_name == Some("start".to_string()));
    assert!(method.is_some());
    assert_eq!(method.unwrap().symbol_type, Some(SymbolType::Method));
}

#[test]
fn typescript_interface_and_abstract_methods() {
    let source = r#"interface Store {
    get(key: string): string | undefined;
    set(key: string, value: string): void;
}

abstract class Base {
    abstract render(): void;
}
"#;
    let chunks = parse(source, "store.ts", Language::TypeScript);

    for name in ["get", "set", "render"] {
        let chunk = chunks
            .iter()
            .find(|c| c.symbol_name == Some(name.to_string()));
        assert!(chunk.is_some(), "missing method chunk: {name}");
        assert_eq!(chunk.unwrap().symbol_type, Some(SymbolType::Method));
    }
}

#[test]
fn typescript_enum_and_type_alias() {
    let source = r#"enum Direction {
    Up,
    Down,
    Left,
    Right,
}

type Point = { x: number; y: number };
"#;
    let chunks = parse(source, "types.ts", Language::TypeScript);

    let enm = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Enum));
    assert!(enm.is_some(), "should have enum chunk");

    let ta = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::TypeAlias));
    assert!(ta.is_some(), "should have type alias chunk");
}

#[test]
fn typescript_const_assigned_functions_are_named() {
    let source = r#"export function declaredFn(a: number): number { return a; }
export const arrowFn = (a: number): number => a;
export const typedArrow: (a: number) => number = (a) => a;
export const asyncArrow = async () => {};
export const exprFn = function (a: number) { return a; };
export const plainConst = 42;
"#;
    let chunks = parse(source, "shapes.ts", Language::TypeScript);

    for name in [
        "declaredFn",
        "arrowFn",
        "typedArrow",
        "asyncArrow",
        "exprFn",
    ] {
        assert_eq!(
            find_chunk(&chunks, name).symbol_type,
            Some(SymbolType::Function),
            "{name} binds a function"
        );
    }

    // Whether plain value consts should be indexed as named symbols is a
    // separate call; this fix deliberately leaves them as they were.
    assert!(
        chunks
            .iter()
            .all(|c| c.symbol_name.as_deref() != Some("plainConst")),
        "value consts should keep their existing shape"
    );

    // The span covers the declaration, initialiser included, not just the name.
    // `export` is present here because chunk content is expanded to whole lines
    // and this declaration is on one line; see the multiline case below.
    assert_eq!(
        find_chunk(&chunks, "arrowFn").content.trim(),
        "export const arrowFn = (a: number): number => a;"
    );
}

/// A declaration split across lines from its `export` keeps the same span shape
/// as every other declaration kind: the symbol covers the declaration, and the
/// `export` on the preceding line falls outside it.
///
/// Pinned deliberately. Const bindings must not become the only declaration
/// kind whose span swallows `export`; changing that is a repo-wide decision
/// about `export_statement`, not something to do for one node kind.
#[test]
fn typescript_multiline_export_spans_match_function_declarations() {
    let source = r#"export
const arrowFn = (a: number): number => a;

export
function declaredFn(a: number): number { return a; }
"#;
    let chunks = parse(source, "multiline.ts", Language::TypeScript);

    for name in ["arrowFn", "declaredFn"] {
        let content = find_chunk(&chunks, name).content.trim().to_string();
        assert!(
            !content.starts_with("export"),
            "{name} should span the declaration, not the export: {content:?}"
        );
    }
}

/// `let` and `var` function bindings are named too. The reported symptom was
/// about `const` because that is the dominant style, but the cause is the name
/// living on the declarator, which is identical for all three keywords.
#[test]
fn javascript_let_var_and_generator_bindings_are_named() {
    let source = r#"export const arrowFn = (a) => a;
let laterFn = function (a) { return a; };
var oldStyleFn = () => {};
export const genFn = function* () { yield 1; };
"#;
    let chunks = parse(source, "shapes.js", Language::JavaScript);

    for name in ["arrowFn", "laterFn", "oldStyleFn", "genFn"] {
        assert_eq!(
            find_chunk(&chunks, name).symbol_type,
            Some(SymbolType::Function),
            "{name} binds a function"
        );
    }
}

#[test]
fn typescript_multi_declarator_statements_are_unchanged() {
    let source = "const a = () => 1, b = () => 2;\n";
    let chunks = parse(source, "multi.ts", Language::TypeScript);

    // The whole shape is pinned, not just the absence of names: one unnamed
    // variable chunk spanning the statement, exactly as before this change.
    assert_eq!(chunks.len(), 1, "unexpected chunks: {chunks:?}");
    assert_eq!(chunks[0].symbol_name, None);
    assert_eq!(chunks[0].symbol_type, Some(SymbolType::Variable));
    assert_eq!(chunks[0].content.trim(), "const a = () => 1, b = () => 2;");
}

#[test]
fn tsx_files_parse_clean_and_expose_jsx_call_sites() {
    let source = r#"import { Hello } from "./hello";

export function App() {
    return <Hello name="world" />;
}
"#;
    let (chunks, refs, diagnostics) = parse_file_with_diagnostics(
        source,
        "src/app.tsx",
        Language::TypeScript,
        &default_config(),
    )
    .unwrap();

    assert!(
        !diagnostics.tree_has_error,
        "the tsx grammar should parse JSX without error nodes"
    );
    assert!(!chunks.is_empty(), "should produce chunks");
    assert!(
        refs.iter().any(|r| r.callee == "Hello"),
        "JSX usage should be recorded as a call site: {refs:?}"
    );
}

// =========================================================
// Go tests
// =========================================================

#[test]
fn go_function_struct_method() {
    let source = r#"package main

import "fmt"

func Hello() string {
    return "hello"
}

type Point struct {
    X float64
    Y float64
}

func (p *Point) Distance() float64 {
    return p.X*p.X + p.Y*p.Y
}
"#;
    let chunks = parse(source, "main.go", Language::Go);
    assert!(
        chunks.len() >= 3,
        "expected func + struct + method, got {}",
        chunks.len()
    );

    let func = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Hello".to_string()));
    assert!(func.is_some());
    assert_eq!(func.unwrap().symbol_type, Some(SymbolType::Function));

    let struc = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Point".to_string()));
    assert!(struc.is_some());
    assert_eq!(struc.unwrap().symbol_type, Some(SymbolType::Struct));

    let method = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Distance".to_string()));
    assert!(method.is_some());
    assert_eq!(method.unwrap().symbol_type, Some(SymbolType::Method));
}

#[test]
fn go_interface() {
    let source = r#"package shapes

type Shape interface {
    Area() float64
    Perimeter() float64
}
"#;
    let chunks = parse(source, "shapes.go", Language::Go);
    let iface = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Interface));
    assert!(iface.is_some(), "should have interface chunk");
    assert_eq!(iface.unwrap().symbol_name, Some("Shape".to_string()));
}

// =========================================================
// Java tests
// =========================================================

#[test]
fn java_class_with_methods() {
    let source = r#"class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"#;
    let chunks = parse(source, "Calculator.java", Language::Java);
    assert!(!chunks.is_empty());

    let cls = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Class));
    assert!(cls.is_some(), "should have class chunk");
}

#[test]
fn java_interface_and_enum() {
    let source = r#"interface Drawable {
    void draw();
    double area();
}

enum Color {
    RED, GREEN, BLUE;
}
"#;
    let chunks = parse(source, "Drawable.java", Language::Java);

    let iface = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Interface));
    assert!(iface.is_some(), "should have interface chunk");

    let enm = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Enum));
    assert!(enm.is_some(), "should have enum chunk");
}

// =========================================================
// C tests
// =========================================================

#[test]
fn c_function_and_struct() {
    let source = r#"#include <stdio.h>

struct Point {
    double x;
    double y;
};

double distance(struct Point* p) {
    return p->x * p->x + p->y * p->y;
}

int main() {
    struct Point p = {3.0, 4.0};
    printf("%f\n", distance(&p));
    return 0;
}
"#;
    let chunks = parse(source, "main.c", Language::C);
    assert!(chunks.len() >= 2, "expected struct + functions");

    let func = chunks
        .iter()
        .find(|c| c.symbol_name == Some("distance".to_string()));
    assert!(func.is_some(), "should find function 'distance'");
    assert_eq!(func.unwrap().symbol_type, Some(SymbolType::Function));
}

// =========================================================
// C++ tests
// =========================================================

#[test]
fn cpp_class_and_namespace() {
    let source = r#"class Shape {
public:
    virtual double area() = 0;
    virtual ~Shape() = default;
};

namespace geometry {
    double pi() {
        return 3.14159;
    }
}
"#;
    let chunks = parse(source, "shape.cpp", Language::Cpp);
    assert!(
        !chunks.is_empty(),
        "expected at least class or namespace chunks"
    );

    let cls = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Class));
    assert!(cls.is_some(), "should have class chunk");

    let ns = chunks
        .iter()
        .find(|c| c.symbol_type == Some(SymbolType::Module));
    assert!(ns.is_some(), "should have namespace chunk");
}

// =========================================================
// Tier 0 fallback tests
// =========================================================

#[test]
fn unknown_language_uses_tier0() {
    let source = "some content\nmore content\nthird line\n";
    let chunks = parse(source, "data.xyz", Language::Unknown);
    assert!(!chunks.is_empty(), "tier0 should produce chunks");
    assert_eq!(chunks[0].language, Language::Unknown);
    assert_eq!(chunks[0].symbol_type, Some(SymbolType::Block));
}

#[test]
fn haskell_indexing_uses_tier0_fallback() {
    let source = "data Point = Point Float Float\nadd x y = x + y\n";
    let (chunks, refs, diagnostics) = parse_file_with_diagnostics(
        source,
        "Data/Point.hs",
        Language::Haskell,
        &default_config(),
    )
    .unwrap();

    assert!(!chunks.is_empty(), "tier0 should produce Haskell chunks");
    assert!(
        refs.is_empty(),
        "tier0 fallback should not extract references"
    );
    assert!(diagnostics.used_tier0_fallback);
}

#[test]
fn toml_uses_whole_file_chunking() {
    let source = "[package]\nname = \"vera\"\nversion = \"0.1.0\"\n";
    let chunks = parse(source, "Cargo.toml", Language::Toml);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].language, Language::Toml);
    assert_eq!(chunks[0].symbol_name.as_deref(), Some("Cargo.toml"));
    assert_eq!(chunks[0].line_start, 1);
    assert_eq!(chunks[0].line_end, 3);
}

#[test]
fn whole_file_chunking_respects_max_chunk_bytes() {
    let source = (0..250)
        .map(|i| format!("key_{i} = \"{}\"", "x".repeat(24)))
        .collect::<Vec<_>>()
        .join("\n");
    let config = IndexingConfig {
        max_chunk_bytes: 600,
        ..Default::default()
    };

    let chunks = parse_and_chunk(&source, "large.toml", Language::Toml, &config).unwrap();

    assert!(
        chunks.len() > 1,
        "oversized whole-file chunks should be split"
    );
    assert!(
        chunks.iter().all(|chunk| chunk.content.len() <= 600),
        "all split chunks should stay within max_chunk_bytes"
    );
}

#[test]
fn rst_respects_max_chunk_bytes() {
    let source = (0..250)
        .map(|i| format!("Paragraph {i}: {}", "x".repeat(28)))
        .collect::<Vec<_>>()
        .join("\n");
    let config = IndexingConfig {
        max_chunk_bytes: 600,
        ..Default::default()
    };

    let chunks = parse_and_chunk(&source, "guide.rst", Language::Rst, &config).unwrap();

    assert!(chunks.len() > 1, "oversized RST chunks should be split");
    assert!(
        chunks.iter().all(|chunk| chunk.content.len() <= 600),
        "all split chunks should stay within max_chunk_bytes"
    );
}

#[test]
fn rst_chunks_are_section_aware() {
    let source = r#"Messenger
=========

Intro paragraph.

Installation
------------

Install the package.

Usage
-----

Dispatch messages.
"#;

    let chunks = parse(source, "guide.rst", Language::Rst);

    let names: Vec<String> = chunks
        .iter()
        .filter_map(|c| c.symbol_name.clone())
        .collect();
    assert!(names.iter().any(|n| n == "Messenger"));
    assert!(names.iter().any(|n| n == "Installation"));
    assert!(names.iter().any(|n| n == "Usage"));

    let usage = chunks
        .iter()
        .find(|chunk| chunk.symbol_name.as_deref() == Some("Usage"))
        .expect("expected a Usage section chunk");
    assert!(usage.content.contains("Dispatch messages."));
}

// =========================================================
// Large symbol splitting
// =========================================================

#[test]
fn large_function_splits_no_content_gaps() {
    // Generate a 500+ line function
    let mut lines = vec!["fn huge_function() {".to_string()];
    for i in 0..550 {
        lines.push(format!("    let x{i} = {i};"));
    }
    lines.push("}".to_string());
    let source = lines.join("\n");

    let config = IndexingConfig {
        max_chunk_lines: 200,
        ..Default::default()
    };
    let chunks = parse_and_chunk(&source, "big.rs", Language::Rust, &config).unwrap();

    // Should be split into multiple sub-chunks
    assert!(chunks.len() >= 3, "552 lines / 200 = 3 sub-chunks expected");

    // Verify no content gaps: concatenating all chunks = original source
    let func_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.symbol_type == Some(SymbolType::Function))
        .collect();
    let mut reconstructed = String::new();
    for (i, chunk) in func_chunks.iter().enumerate() {
        if i > 0 {
            reconstructed.push('\n');
        }
        reconstructed.push_str(&chunk.content);
    }
    assert_eq!(
        reconstructed, source,
        "reconstructed content should match original"
    );

    // Verify line ranges are contiguous
    for i in 1..func_chunks.len() {
        assert_eq!(
            func_chunks[i].line_start,
            func_chunks[i - 1].line_end + 1,
            "sub-chunks should be contiguous"
        );
    }
}

// =========================================================
// Metadata correctness
// =========================================================

#[test]
fn chunk_metadata_matches_source() {
    let source = r#"fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}"#;
    let chunks = parse(source, "lib.rs", Language::Rust);
    assert_eq!(chunks.len(), 1);

    let chunk = &chunks[0];
    assert_eq!(chunk.file_path, "lib.rs");
    assert_eq!(chunk.language, Language::Rust);
    assert_eq!(chunk.symbol_type, Some(SymbolType::Function));
    assert_eq!(chunk.symbol_name, Some("greet".to_string()));
    assert_eq!(chunk.line_start, 1);
    assert_eq!(chunk.line_end, 3);

    // Verify content matches source at declared lines
    let source_lines: Vec<&str> = source.lines().collect();
    let expected = source_lines[0..3].join("\n");
    assert_eq!(chunk.content, expected);
}

#[test]
fn javascript_extracts_like_typescript() {
    let source = r#"function sum(a, b) {
    return a + b;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
"#;
    let chunks = parse(source, "calc.js", Language::JavaScript);
    assert!(chunks.len() >= 2, "expected function + class");

    let func = chunks
        .iter()
        .find(|c| c.symbol_name == Some("sum".to_string()));
    assert!(func.is_some());

    let cls = chunks
        .iter()
        .find(|c| c.symbol_name == Some("Calculator".to_string()));
    assert!(cls.is_some());
}

// =========================================================
// Edge cases
// =========================================================

#[test]
fn empty_source_produces_no_chunks() {
    let chunks = parse_and_chunk("", "empty.rs", Language::Rust, &default_config()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn whitespace_only_source_no_chunks() {
    let chunks =
        parse_and_chunk("   \n\n  \n", "ws.rs", Language::Rust, &default_config()).unwrap();
    assert!(chunks.is_empty());
}

#[test]
fn comment_only_file_falls_back() {
    let source = "// This is a comment\n// Another comment\n";
    let chunks = parse(source, "comments.rs", Language::Rust);
    // Should produce Tier 0 fallback chunks since no symbols found
    assert!(
        !chunks.is_empty(),
        "comment-only files should still produce chunks"
    );
}
