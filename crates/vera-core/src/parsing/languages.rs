//! Tree-sitter grammar loading for supported languages.
//!
//! Maps [`Language`] variants to tree-sitter grammar definitions.
//! Tier 1A languages get full AST-based parsing; others fall back to Tier 0.

use tree_sitter::Language as TsLanguage;

use crate::types::Language;

extern crate tree_sitter_hcl;

unsafe extern "C" {
    fn tree_sitter_sql() -> *const ();
    fn tree_sitter_hcl() -> *const ();
    fn tree_sitter_proto() -> *const ();
    fn tree_sitter_scss() -> *const ();
    fn tree_sitter_vue() -> *const ();
    fn tree_sitter_dockerfile() -> *const ();
    fn tree_sitter_astro() -> *const ();
}

/// Returns the tree-sitter grammar for a given language, if supported.
///
/// Returns `None` for languages without tree-sitter support (Tier 0 fallback).
pub fn tree_sitter_grammar(lang: Language) -> Option<TsLanguage> {
    let lang_fn = match lang {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        Language::Bash => tree_sitter_bash::LANGUAGE.into(),
        Language::Kotlin => tree_sitter_kotlin_sg::LANGUAGE.into(),
        Language::Swift => tree_sitter_swift::LANGUAGE.into(),
        Language::Zig => tree_sitter_zig::LANGUAGE.into(),
        Language::Lua => tree_sitter_lua::LANGUAGE.into(),
        Language::Scala => tree_sitter_scala::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Php => tree_sitter_php::LANGUAGE_PHP.into(),
        Language::Haskell => tree_sitter_haskell::LANGUAGE.into(),
        Language::Elixir => tree_sitter_elixir::LANGUAGE.into(),
        Language::Dart => tree_sitter_dart::LANGUAGE.into(),
        Language::Sql => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_sql()) },
        Language::Hcl => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_hcl()) },
        Language::Protobuf => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_proto())
        },
        Language::Html => tree_sitter_html::LANGUAGE.into(),
        Language::Css => tree_sitter_css::LANGUAGE.into(),
        Language::Scss => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_scss())
        },
        Language::Vue => unsafe { std::mem::transmute::<*const (), TsLanguage>(tree_sitter_vue()) },
        Language::GraphQl => tree_sitter_graphql::LANGUAGE.into(),
        Language::CMake => tree_sitter_cmake::LANGUAGE.into(),
        Language::Dockerfile => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_dockerfile())
        },
        Language::Xml => tree_sitter_xml::LANGUAGE_XML.into(),
        // Tier 2A code languages
        Language::ObjectiveC => tree_sitter_objc::LANGUAGE.into(),
        Language::Perl => tree_sitter_perl::LANGUAGE.into(),
        Language::Julia => tree_sitter_julia::LANGUAGE.into(),
        Language::Nix => tree_sitter_nix::LANGUAGE.into(),
        Language::OCaml => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        Language::Groovy => tree_sitter_groovy::LANGUAGE.into(),
        Language::Clojure => tree_sitter_clojure_orchard::LANGUAGE.into(),
        Language::CommonLisp => tree_sitter_commonlisp::LANGUAGE_COMMONLISP.into(),
        Language::Erlang => tree_sitter_erlang::LANGUAGE.into(),
        Language::FSharp => tree_sitter_fsharp::LANGUAGE_FSHARP.into(),
        Language::Fortran => tree_sitter_fortran::LANGUAGE.into(),
        Language::PowerShell => tree_sitter_powershell::LANGUAGE.into(),
        Language::R => tree_sitter_r::LANGUAGE.into(),
        // Tier 2A batch 2 code languages
        Language::Matlab => tree_sitter_matlab::LANGUAGE.into(),
        Language::DLang => tree_sitter_d::LANGUAGE.into(),
        Language::Fish => tree_sitter_fish::language(),
        Language::Zsh => tree_sitter_zsh::LANGUAGE.into(),
        Language::Luau => tree_sitter_luau::LANGUAGE.into(),
        Language::Scheme => tree_sitter_scheme::LANGUAGE.into(),
        Language::Racket => tree_sitter_racket::LANGUAGE.into(),
        Language::Elm => tree_sitter_elm::LANGUAGE.into(),
        Language::Glsl => tree_sitter_glsl::LANGUAGE_GLSL.into(),
        Language::Hlsl => tree_sitter_hlsl::LANGUAGE_HLSL.into(),
        // Tier 2B structural/config/frontend languages
        Language::Svelte => tree_sitter_svelte_next::LANGUAGE.into(),
        Language::Astro => unsafe {
            std::mem::transmute::<*const (), TsLanguage>(tree_sitter_astro())
        },
        Language::Makefile => tree_sitter_make::LANGUAGE.into(),
        Language::Ini => tree_sitter_ini::LANGUAGE.into(),
        Language::Nginx => tree_sitter_nginx::LANGUAGE.into(),
        Language::Prisma => tree_sitter_prisma_io::LANGUAGE.into(),
        Language::Rst => tree_sitter_rst::LANGUAGE.into(),
        // Languages without tree-sitter grammar support → Tier 0 fallback
        Language::Toml
        | Language::Yaml
        | Language::Json
        | Language::Markdown
        | Language::Unknown => return None,
    };
    Some(lang_fn)
}

/// Returns the tree-sitter grammar to use for a specific file.
///
/// TypeScript ships as two grammars upstream because JSX is ambiguous with
/// type assertions, so `.tsx` must be parsed with `tsx`. Parsing it with
/// `typescript` puts every JSX element in an error node, which hides JSX call
/// sites from `references` and `dead-code`. All other languages resolve by
/// language alone.
pub fn tree_sitter_grammar_for_path(lang: Language, file_path: &str) -> Option<TsLanguage> {
    if lang == Language::TypeScript && has_tsx_extension(file_path) {
        return Some(tree_sitter_typescript::LANGUAGE_TSX.into());
    }
    tree_sitter_grammar(lang)
}

fn has_tsx_extension(file_path: &str) -> bool {
    std::path::Path::new(file_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tsx"))
}

/// Returns whether a language has tree-sitter grammar support (Tier 1A).
pub fn has_grammar(lang: Language) -> bool {
    tree_sitter_grammar(lang).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_1a_languages_have_grammars() {
        let tier_1a = [
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Python,
            Language::Go,
            Language::Java,
            Language::C,
            Language::Cpp,
            Language::Ruby,
            Language::Bash,
            Language::Kotlin,
            Language::Swift,
            Language::Zig,
            Language::Lua,
            Language::Scala,
            Language::CSharp,
            Language::Php,
            Language::Haskell,
            Language::Elixir,
            Language::Dart,
            Language::Sql,
            Language::Hcl,
            Language::Protobuf,
        ];
        for lang in tier_1a {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn tier_1b_languages_have_grammars() {
        let tier_1b = [
            Language::Html,
            Language::Css,
            Language::Scss,
            Language::Vue,
            Language::GraphQl,
            Language::CMake,
            Language::Dockerfile,
            Language::Xml,
        ];
        for lang in tier_1b {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    #[test]
    fn tier_0_languages_have_no_grammar() {
        let tier_0 = [
            Language::Unknown,
            Language::Toml,
            Language::Yaml,
            Language::Json,
            Language::Markdown,
        ];
        for lang in tier_0 {
            assert!(
                !has_grammar(lang),
                "{lang} should NOT have a tree-sitter grammar"
            );
        }
    }

    /// A `.tsx` file must parse without error nodes; the `typescript` grammar
    /// puts every JSX element in one, which is what hides JSX call sites.
    #[test]
    fn tsx_files_parse_jsx_without_errors() {
        let source = "export function Hello() {\n  return <div className=\"greet\">hi</div>;\n}\n";

        let tsx = tree_sitter_grammar_for_path(Language::TypeScript, "src/jsx.tsx").unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&tsx).unwrap();
        assert!(
            !parser.parse(source, None).unwrap().root_node().has_error(),
            "tsx grammar should parse JSX cleanly"
        );

        let ts = tree_sitter_grammar_for_path(Language::TypeScript, "src/plain.ts").unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts).unwrap();
        assert!(
            parser.parse(source, None).unwrap().root_node().has_error(),
            "typescript grammar is the one that errors on JSX; if this stops \
             holding, the tsx split is no longer needed"
        );
    }

    #[test]
    fn non_tsx_paths_keep_their_language_grammar() {
        let cases = [
            ("src/plain.ts", Language::TypeScript),
            ("src/app.mts", Language::TypeScript),
            ("noext", Language::TypeScript),
            ("src/main.rs", Language::Rust),
        ];
        for (path, lang) in cases {
            let by_path = tree_sitter_grammar_for_path(lang, path).unwrap();
            assert_eq!(
                by_path,
                tree_sitter_grammar(lang).unwrap(),
                "{path} should use the plain {lang} grammar"
            );
        }
    }

    #[test]
    fn grammar_creates_valid_parser() {
        let grammar = tree_sitter_grammar(Language::Rust).unwrap();
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&grammar).expect("grammar should load");
    }

    /// Every shipped grammar must parse a small representative sample
    /// without error nodes.
    #[test]
    fn grammars_parse_samples_without_errors() {
        let cases: &[(Language, &str)] = &[
            (Language::Html, "<div></div>"),
            (Language::Css, "body { color: red; }"),
            (Language::Scss, "$color: red; body { color: $color; }"),
            (Language::Vue, "<template><div></div></template>"),
            (Language::GraphQl, "type Query { hello: String }"),
            (Language::CMake, "cmake_minimum_required(VERSION 3.10)"),
            (Language::Dockerfile, "FROM ubuntu:20.04\n"),
            (Language::Xml, "<?xml version=\"1.0\"?><root><item/></root>"),
            (
                Language::ObjectiveC,
                "@interface Foo : NSObject\n- (void)bar;\n@end\n",
            ),
            (Language::Perl, "sub hello { print \"hello\\n\"; }\n"),
            (
                Language::Julia,
                "function hello()\n println(\"hello\")\nend\n",
            ),
            (
                Language::Nix,
                "{ pkgs ? import <nixpkgs> {} }: pkgs.hello\n",
            ),
            (Language::OCaml, "let hello () = print_endline \"hello\"\n"),
            (Language::Groovy, "def hello() { println 'hello' }\n"),
            (Language::Clojure, "(defn hello [] (println \"hello\"))\n"),
            (
                Language::CommonLisp,
                "(defun hello () (format t \"hello~%\"))\n",
            ),
            (Language::Erlang, "-module(hello).\nhello() -> ok.\n"),
            (Language::FSharp, "let hello () = printfn \"hello\"\n"),
            (
                Language::Fortran,
                "program hello\n print *, 'hello'\nend program hello\n",
            ),
            (
                Language::PowerShell,
                "function Hello { Write-Host 'hello' }\n",
            ),
            (Language::R, "hello <- function() { print(\"hello\") }\n"),
            (Language::Matlab, "function y = square(x)\n y = x^2;\nend\n"),
            (Language::DLang, "void main() { writeln(\"hello\"); }\n"),
            (Language::Fish, "function hello\n echo hello\nend\n"),
            (Language::Zsh, "function hello() {\n echo hello\n}\n"),
            (
                Language::Luau,
                "local function hello()\n print(\"hello\")\nend\n",
            ),
            (Language::Scheme, "(define (hello) (display \"hello\"))\n"),
            (
                Language::Racket,
                "#lang racket\n(define (hello) (displayln \"hello\"))\n",
            ),
            (
                Language::Elm,
                "module Main exposing (main)\n\nmain =\n text \"hello\"\n",
            ),
            (
                Language::Glsl,
                "void main() {\n gl_FragColor = vec4(1.0);\n}\n",
            ),
            (
                Language::Hlsl,
                "float4 main(float4 pos : SV_Position) : SV_Target {\n return float4(1, 0, 0, 1);\n}\n",
            ),
            (
                Language::Svelte,
                "<script>\n let count = 0;\n</script>\n<button>{count}</button>\n",
            ),
            (
                Language::Astro,
                "---\nconst title = \"Hello\";\n---\n<h1>{title}</h1>\n",
            ),
            (
                Language::Makefile,
                "all: build\n\nbuild:\n\tgcc -o main main.c\n",
            ),
            (Language::Ini, "[section]\nkey = value\n"),
            (
                Language::Nginx,
                "server {\n listen 80;\n server_name example.com;\n}\n",
            ),
            (
                Language::Prisma,
                "model User {\n id Int @id @default(autoincrement())\n name String\n}\n",
            ),
            (
                Language::Rst,
                "Heading\n=======\n\nParagraph text.\n\nSection\n-------\n\nMore text.\n",
            ),
        ];
        for (lang, sample) in cases {
            let grammar = tree_sitter_grammar(*lang).unwrap();
            let mut parser = tree_sitter::Parser::new();
            parser
                .set_language(&grammar)
                .unwrap_or_else(|_| panic!("{lang} grammar should load"));
            let tree = parser.parse(sample, None).unwrap();
            assert!(
                !tree.root_node().has_error(),
                "{lang} sample should parse without errors"
            );
        }
    }

    // ── Tier 2A grammar loading tests ─────────────────────────

    #[test]
    fn tier_2a_languages_have_grammars() {
        let tier_2a = [
            Language::ObjectiveC,
            Language::Perl,
            Language::Julia,
            Language::Nix,
            Language::OCaml,
            Language::Groovy,
            Language::Clojure,
            Language::CommonLisp,
            Language::Erlang,
            Language::FSharp,
            Language::Fortran,
            Language::PowerShell,
            Language::R,
        ];
        for lang in tier_2a {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    // ── Tier 2A batch 2 grammar loading tests ─────────────────

    #[test]
    fn tier_2a_batch2_languages_have_grammars() {
        let tier_2a_b2 = [
            Language::Matlab,
            Language::DLang,
            Language::Fish,
            Language::Zsh,
            Language::Luau,
            Language::Scheme,
            Language::Racket,
            Language::Elm,
            Language::Glsl,
            Language::Hlsl,
        ];
        for lang in tier_2a_b2 {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }

    // ── Tier 2B grammar loading tests ─────────────────

    #[test]
    fn tier_2b_languages_have_grammars() {
        let tier_2b = [
            Language::Svelte,
            Language::Astro,
            Language::Makefile,
            Language::Ini,
            Language::Nginx,
            Language::Prisma,
            Language::Rst,
        ];
        for lang in tier_2b {
            assert!(
                has_grammar(lang),
                "{lang} should have a tree-sitter grammar"
            );
        }
    }
}
