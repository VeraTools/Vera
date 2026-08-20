//! Container declarations must not swallow the symbols declared inside them.
//!
//! One fixture per language whose grammar classifies a body-spanning container
//! (class, module, namespace, trait, object). Each asserts the inner symbols
//! are extracted with the right type *and* that the container is still
//! recorded, so a regression in either direction fails.

use super::*;
use crate::parsing::languages::tree_sitter_grammar;

fn extracted(source: &str, lang: Language) -> Vec<RawSymbol> {
    let grammar = tree_sitter_grammar(lang).unwrap();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&grammar).unwrap();
    let tree = parser.parse(source, None).unwrap();
    extract_symbols(&tree, source.as_bytes(), lang)
}

fn describe(symbols: &[RawSymbol]) -> String {
    symbols
        .iter()
        .map(|s| format!("{:?} {:?} @{}", s.symbol_type, s.name, s.start_row))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Require a symbol with exactly this name and type.
fn require<'a>(symbols: &'a [RawSymbol], name: &str, ty: SymbolType) -> &'a RawSymbol {
    symbols
        .iter()
        .find(|s| s.name.as_deref() == Some(name) && s.symbol_type == ty)
        .unwrap_or_else(|| panic!("expected {ty:?} {name:?}, got [{}]", describe(symbols)))
}

/// Require a symbol of this type starting on this row. Used where the grammar
/// gives the extractor no name to key on; the row still pins one symbol rather
/// than counting them.
fn require_at(symbols: &[RawSymbol], start_row: usize, ty: SymbolType) -> &RawSymbol {
    symbols
        .iter()
        .find(|s| s.start_row == start_row && s.symbol_type == ty)
        .unwrap_or_else(|| {
            panic!(
                "expected {ty:?} at row {start_row}, got [{}]",
                describe(symbols)
            )
        })
}

/// Require exactly one symbol of this type, catching a container recorded twice
/// (its own node plus a header child or a bare keyword token).
fn require_only(symbols: &[RawSymbol], ty: SymbolType) -> &RawSymbol {
    let mut matching = symbols.iter().filter(|s| s.symbol_type == ty);
    let first = matching
        .next()
        .unwrap_or_else(|| panic!("expected one {ty:?}, got [{}]", describe(symbols)));
    if let Some(extra) = matching.next() {
        panic!(
            "expected one {ty:?}, also got {:?} @{}: [{}]",
            extra.name,
            extra.start_row,
            describe(symbols)
        );
    }
    first
}

/// The inner symbol must sit strictly inside the container's span, which is
/// what makes it a symbol the container used to swallow.
fn assert_nested(outer: &RawSymbol, inner: &RawSymbol) {
    assert!(
        inner.start_byte > outer.start_byte && inner.end_byte <= outer.end_byte,
        "{:?} {:?} ({}..{}) should be nested inside {:?} {:?} ({}..{})",
        inner.symbol_type,
        inner.name,
        inner.start_byte,
        inner.end_byte,
        outer.symbol_type,
        outer.name,
        outer.start_byte,
        outer.end_byte,
    );
}

#[test]
fn ruby_module_and_class_keep_their_members() {
    let symbols = extracted(
        r#"
module Outer
  class Widget
    def render
      1
    end
  end

  def helper
    2
  end
end
"#,
        Language::Ruby,
    );

    let module = require(&symbols, "Outer", SymbolType::Module);
    let class = require_only(&symbols, SymbolType::Class);
    assert_eq!(class.name.as_deref(), Some("Widget"));
    let render = require(&symbols, "render", SymbolType::Function);
    let helper = require(&symbols, "helper", SymbolType::Function);

    assert_nested(module, class);
    assert_nested(class, render);
    assert_nested(module, helper);
}

#[test]
fn kotlin_class_and_object_keep_their_members() {
    let symbols = extracted(
        r#"
class Widget {
    fun render(): Int = 1
}

object Registry {
    fun lookup(): Int = 2
}
"#,
        Language::Kotlin,
    );

    let class = require(&symbols, "Widget", SymbolType::Class);
    let object = require(&symbols, "Registry", SymbolType::Class);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(object, require(&symbols, "lookup", SymbolType::Function));
}

#[test]
fn swift_class_and_struct_keep_their_members() {
    let symbols = extracted(
        r#"
class Widget {
    func render() -> Int { return 1 }
}

struct Point {
    func norm() -> Int { return 2 }
}
"#,
        Language::Swift,
    );

    // The refined type survives the recursion: a struct stays a struct.
    let class = require(&symbols, "Widget", SymbolType::Class);
    let structure = require(&symbols, "Point", SymbolType::Struct);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(structure, require(&symbols, "norm", SymbolType::Function));
}

#[test]
fn scala_class_trait_and_object_keep_their_members() {
    let symbols = extracted(
        r#"
class Widget {
  def render(): Int = 1
}

trait Drawable {
  def draw(): Int = 2
}

object Registry {
  def lookup(): Int = 3
}
"#,
        Language::Scala,
    );

    let class = require(&symbols, "Widget", SymbolType::Class);
    let sealed_trait = require(&symbols, "Drawable", SymbolType::Trait);
    let object = require(&symbols, "Registry", SymbolType::Module);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(
        sealed_trait,
        require(&symbols, "draw", SymbolType::Function),
    );
    assert_nested(object, require(&symbols, "lookup", SymbolType::Function));
}

#[test]
fn cpp_namespace_and_class_keep_their_members() {
    let symbols = extracted(
        r#"
namespace outer {

class Widget {
public:
  int render() { return 1; }
};

int helper() { return 2; }

}
"#,
        Language::Cpp,
    );

    // The namespace node carries no name the extractor can read, so pin it by
    // its span instead.
    let namespace = require_at(&symbols, 1, SymbolType::Module);
    let class = require(&symbols, "Widget", SymbolType::Class);
    let render = require(&symbols, "render", SymbolType::Function);
    let helper = require(&symbols, "helper", SymbolType::Function);

    assert_nested(namespace, class);
    assert_nested(class, render);
    assert_nested(namespace, helper);
}

#[test]
fn groovy_class_and_interface_keep_their_members() {
    let symbols = extracted(
        r#"
class Widget {
    def render() { return 1 }
}

interface Drawable {
    def draw()
}
"#,
        Language::Groovy,
    );

    let class = require(&symbols, "Widget", SymbolType::Class);
    let interface = require(&symbols, "Drawable", SymbolType::Interface);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(interface, require(&symbols, "draw", SymbolType::Function));
}

#[test]
fn powershell_class_keeps_its_methods() {
    let symbols = extracted(
        r#"
class Widget {
    [int] Render() { return 1 }
    static [int] Build() { return 2 }
}
"#,
        Language::PowerShell,
    );

    let class = require_at(&symbols, 1, SymbolType::Class);
    assert_nested(class, require(&symbols, "Render", SymbolType::Method));
    assert_nested(class, require(&symbols, "Build", SymbolType::Method));
}

#[test]
fn matlab_classdef_keeps_its_methods() {
    let symbols = extracted(
        r#"
classdef Widget
    properties
        a
    end
    methods
        function y = render(obj)
            y = 1;
        end
        function z = build(obj)
            z = 2;
        end
    end
end
"#,
        Language::Matlab,
    );

    let class = require(&symbols, "Widget", SymbolType::Class);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(class, require(&symbols, "build", SymbolType::Function));
}

#[test]
fn dlang_class_and_struct_keep_their_members() {
    let symbols = extracted(
        r#"
module app;

class Widget {
    int render() { return 1; }
}

struct Holder {
    int fetch() { return 2; }
}
"#,
        Language::DLang,
    );

    // `module app;` is a header, not a container: it must stay a leaf symbol
    // spanning its own line only.
    let header = require_at(&symbols, 1, SymbolType::Module);
    assert_eq!(header.end_row, 1);

    let class = require(&symbols, "Widget", SymbolType::Class);
    let structure = require(&symbols, "Holder", SymbolType::Struct);
    assert_nested(class, require(&symbols, "render", SymbolType::Function));
    assert_nested(structure, require(&symbols, "fetch", SymbolType::Function));
}

#[test]
fn ocaml_module_keeps_its_definitions() {
    let symbols = extracted(
        r#"
module Outer = struct
  let helper x = x + 1

  type t = int
end
"#,
        Language::OCaml,
    );

    let module = require_at(&symbols, 1, SymbolType::Module);
    assert_nested(module, require_at(&symbols, 2, SymbolType::Function));
    assert_nested(module, require_at(&symbols, 4, SymbolType::TypeAlias));
}

#[test]
fn fsharp_module_keeps_its_definitions() {
    let symbols = extracted(
        r#"
module Outer =
    let helper x = x + 1

    let other y = y - 1
"#,
        Language::FSharp,
    );

    let module = require(&symbols, "Outer", SymbolType::Module);
    assert_nested(module, require_at(&symbols, 2, SymbolType::Function));
    assert_nested(module, require_at(&symbols, 4, SymbolType::Function));
}

#[test]
fn julia_module_keeps_its_definitions() {
    let symbols = extracted(
        r#"
module Outer

function helper(x)
    x + 1
end

struct Widget
    a::Int
end

end
"#,
        Language::Julia,
    );

    let module = require(&symbols, "Outer", SymbolType::Module);
    assert_nested(module, require_at(&symbols, 3, SymbolType::Function));
    assert_nested(module, require_at(&symbols, 7, SymbolType::Struct));
}

#[test]
fn fortran_module_keeps_its_procedures() {
    let symbols = extracted(
        r#"
module widgets
contains
  function helper(x)
    integer :: helper, x
    helper = x + 1
  end function helper
end module widgets
"#,
        Language::Fortran,
    );

    // One module symbol, spanning the whole module: the `module_statement`
    // header must not be recorded as a second one.
    let module = require_only(&symbols, SymbolType::Module);
    assert_eq!(module.start_row, 1);
    assert_nested(module, require_at(&symbols, 3, SymbolType::Function));
}
