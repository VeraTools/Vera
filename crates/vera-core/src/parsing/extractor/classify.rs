//! Tree-sitter node-kind to SymbolType classification tables.

use crate::types::{Language, SymbolType};

pub(crate) fn classify_node(lang: Language, kind: &str) -> Option<SymbolType> {
    match lang {
        Language::Rust => classify_with(RUST_KINDS, kind),
        Language::TypeScript | Language::JavaScript => classify_with(TYPESCRIPT_KINDS, kind),
        Language::Python => classify_with(PYTHON_KINDS, kind),
        Language::Go => classify_with(GO_KINDS, kind),
        Language::Java => classify_with(JAVA_KINDS, kind),
        Language::C => classify_with(C_KINDS, kind),
        Language::Cpp => classify_with(CPP_KINDS, kind),
        Language::Ruby => classify_with(RUBY_KINDS, kind),
        Language::Bash => classify_with(BASH_KINDS, kind),
        Language::Kotlin => classify_with(KOTLIN_KINDS, kind),
        Language::Swift => classify_with(SWIFT_KINDS, kind),
        Language::Zig => classify_with(ZIG_KINDS, kind),
        Language::Lua => classify_with(LUA_KINDS, kind),
        Language::Scala => classify_with(SCALA_KINDS, kind),
        Language::CSharp => classify_with(CSHARP_KINDS, kind),
        Language::Php => classify_with(PHP_KINDS, kind),
        Language::Haskell => classify_with(HASKELL_KINDS, kind),
        Language::Dart => classify_with(DART_KINDS, kind),
        Language::Sql => classify_with(SQL_KINDS, kind),
        Language::Hcl => classify_with(HCL_KINDS, kind),
        Language::Protobuf => classify_with(PROTOBUF_KINDS, kind),
        Language::Html => classify_with(HTML_KINDS, kind),
        Language::Css => classify_with(CSS_KINDS, kind),
        Language::Scss => classify_with(SCSS_KINDS, kind),
        Language::Vue => classify_with(VUE_KINDS, kind),
        Language::GraphQl => classify_with(GRAPHQL_KINDS, kind),
        Language::CMake => classify_with(CMAKE_KINDS, kind),
        Language::Dockerfile => classify_with(DOCKERFILE_KINDS, kind),
        Language::Xml => classify_with(XML_KINDS, kind),
        Language::ObjectiveC => classify_with(OBJECTIVEC_KINDS, kind),
        Language::Perl => classify_with(PERL_KINDS, kind),
        Language::Julia => classify_with(JULIA_KINDS, kind),
        Language::Nix => classify_with(NIX_KINDS, kind),
        Language::OCaml => classify_with(OCAML_KINDS, kind),
        Language::Groovy => classify_with(GROOVY_KINDS, kind),
        Language::Clojure => classify_with(CLOJURE_KINDS, kind),
        Language::CommonLisp => classify_with(COMMONLISP_KINDS, kind),
        Language::Erlang => classify_with(ERLANG_KINDS, kind),
        Language::FSharp => classify_with(FSHARP_KINDS, kind),
        Language::Fortran => classify_with(FORTRAN_KINDS, kind),
        Language::PowerShell => classify_with(POWERSHELL_KINDS, kind),
        Language::R => classify_with(R_KINDS, kind),
        Language::Matlab => classify_with(MATLAB_KINDS, kind),
        Language::DLang => classify_with(DLANG_KINDS, kind),
        Language::Fish => classify_with(FISH_KINDS, kind),
        Language::Zsh => classify_with(ZSH_KINDS, kind),
        Language::Luau => classify_with(LUAU_KINDS, kind),
        Language::Scheme => classify_with(SCHEME_KINDS, kind),
        Language::Racket => classify_with(RACKET_KINDS, kind),
        Language::Elm => classify_with(ELM_KINDS, kind),
        Language::Glsl => classify_with(GLSL_KINDS, kind),
        Language::Hlsl => classify_with(HLSL_KINDS, kind),
        Language::Svelte => classify_with(SVELTE_KINDS, kind),
        Language::Astro => classify_with(ASTRO_KINDS, kind),
        Language::Makefile => classify_with(MAKEFILE_KINDS, kind),
        Language::Ini => classify_with(INI_KINDS, kind),
        Language::Nginx => classify_with(NGINX_KINDS, kind),
        Language::Prisma => classify_with(PRISMA_KINDS, kind),
        _ => None,
    }
}

/// Look up a tree-sitter node kind in a per-language symbol table.
fn classify_with(table: &[(&[&str], SymbolType)], kind: &str) -> Option<SymbolType> {
    table
        .iter()
        .find_map(|(kinds, ty)| kinds.contains(&kind).then_some(*ty))
}

/// Node kinds that [`classify_node`] recognises *and* whose span covers a body
/// holding further symbols.
///
/// The extractor stops at the first classified node, so a container listed here
/// would otherwise swallow everything declared inside it: one chunk for the
/// whole class, module or namespace and no symbol for any method in it. The
/// extractor records the container itself and then keeps walking into it.
///
/// Deliberately absent:
/// - Header nodes that end at the declaration and never span a body: Perl
///   `package_statement`, Erlang `module_attribute`, Elm `module_declaration`,
///   D `module_declaration`, Clojure `ns`, Common Lisp `defpackage`.
/// - Whole-unit chunks whose contents are markup or data rather than callables:
///   CSS `rule_set`, HTML/XML `element`, INI `section`, GraphQL type
///   definitions, Nix `let_expression`, HCL `block`.
/// - Constructs that do span classified children but are neither containers of
///   callables nor part of this change: Protobuf `message` (nested messages and
///   enums) and CMake `if_condition`/`foreach_loop`/`while_loop` (a
///   conditionally defined `function_def`). Both swallow today; both are a
///   different construct family and want their own change.
pub(crate) fn container_body_kinds(lang: Language) -> &'static [&'static str] {
    match lang {
        // `file_scoped_namespace_declaration` ends at its semicolon rather than
        // spanning the file, so recursing into it finds nothing. It is kept
        // because the case this replaced listed it; a test pins that its
        // members are siblings rather than children.
        Language::CSharp => &["namespace_declaration", "file_scoped_namespace_declaration"],
        Language::Ruby => &["class", "module"],
        Language::Kotlin => &["class_declaration", "object_declaration"],
        // Swift spells `struct`, `enum` and `extension` as `class_declaration`
        // too, so all four are covered by the one kind.
        Language::Swift => &["class_declaration"],
        Language::Scala => &["class_definition", "trait_definition", "object_definition"],
        Language::Cpp => &[
            "namespace_definition",
            "class_specifier",
            "struct_specifier",
        ],
        // HLSL is a C++-family grammar and spells member functions the same
        // way. GLSL is not listed: its `struct_specifier` holds no callables.
        Language::Hlsl => &["class_specifier", "struct_specifier"],
        Language::Groovy => &[
            "class_definition",
            "class_declaration",
            "interface_definition",
            "interface_declaration",
        ],
        Language::PowerShell => &["class_statement"],
        Language::Matlab => &["class_definition"],
        Language::DLang => &[
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "template_declaration",
        ],
        Language::OCaml => &["module_definition"],
        Language::FSharp => &["module_defn"],
        Language::Julia => &["module_definition"],
        Language::Fortran => &["module", "program"],
        _ => &[],
    }
}

const SQL_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["create_table", "create_table_statement", "table_definition"],
        SymbolType::Struct,
    ),
    (
        &[
            "create_function",
            "create_function_statement",
            "function_definition",
            "create_procedure_statement",
            "create_procedure",
            "create_view",
            "create_view_statement",
            "view_definition",
        ],
        SymbolType::Function,
    ),
];

const HCL_KINDS: &[(&[&str], SymbolType)] = &[(&["block"], SymbolType::Struct)];

const PROTOBUF_KINDS: &[(&[&str], SymbolType)] = &[
    (&["message", "message_definition"], SymbolType::Struct),
    (&["enum", "enum_definition"], SymbolType::Enum),
    (&["service", "service_definition"], SymbolType::Class),
    (
        &["rpc", "rpc_definition", "rpc_declaration"],
        SymbolType::Method,
    ),
];

const HTML_KINDS: &[(&[&str], SymbolType)] = &[(
    &["element", "script_element", "style_element"],
    SymbolType::Block,
)];

const CSS_KINDS: &[(&[&str], SymbolType)] = &[
    (&["rule_set"], SymbolType::Block),
    (&["media_statement"], SymbolType::Block),
    (&["keyframes_statement"], SymbolType::Block),
    (&["import_statement"], SymbolType::Variable),
];

const SCSS_KINDS: &[(&[&str], SymbolType)] = &[
    (&["rule_set"], SymbolType::Block),
    (&["mixin_statement"], SymbolType::Function),
    (&["function_statement"], SymbolType::Function),
    (&["include_statement"], SymbolType::Variable),
    (&["media_statement"], SymbolType::Block),
    (&["keyframes_statement"], SymbolType::Block),
];

const VUE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["template_element"], SymbolType::Block),
    (&["script_element"], SymbolType::Block),
    (&["style_element"], SymbolType::Block),
];

const GRAPHQL_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["object_type_definition", "input_object_type_definition"],
        SymbolType::Struct,
    ),
    (&["interface_type_definition"], SymbolType::Interface),
    (&["enum_type_definition"], SymbolType::Enum),
    (&["union_type_definition"], SymbolType::TypeAlias),
    (&["scalar_type_definition"], SymbolType::TypeAlias),
    (&["schema_definition"], SymbolType::Block),
    (&["operation_definition"], SymbolType::Function),
    (&["fragment_definition"], SymbolType::Function),
    (&["directive_definition"], SymbolType::Function),
];

const CMAKE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_def", "macro_def"], SymbolType::Function),
    (
        &["if_condition", "foreach_loop", "while_loop"],
        SymbolType::Block,
    ),
    (&["normal_command"], SymbolType::Variable),
];

const DOCKERFILE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["from_instruction"], SymbolType::Block),
    (&["run_instruction"], SymbolType::Block),
    (
        &["copy_instruction", "add_instruction"],
        SymbolType::Variable,
    ),
    (
        &["cmd_instruction", "entrypoint_instruction"],
        SymbolType::Function,
    ),
    (
        &["env_instruction", "arg_instruction", "label_instruction"],
        SymbolType::Constant,
    ),
    (&["expose_instruction"], SymbolType::Variable),
    (
        &[
            "workdir_instruction",
            "user_instruction",
            "volume_instruction",
        ],
        SymbolType::Variable,
    ),
];

const XML_KINDS: &[(&[&str], SymbolType)] = &[(&["element"], SymbolType::Block)];

const OBJECTIVEC_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (
        &[
            "method_declaration",
            "method_definition",
            "implementation_definition",
        ],
        SymbolType::Method,
    ),
    (
        &["class_interface", "class_implementation"],
        SymbolType::Class,
    ),
    (&["protocol_declaration"], SymbolType::Interface),
    (
        &["category_interface", "category_implementation"],
        SymbolType::Class,
    ),
];

const PERL_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_definition", "subroutine_declaration_statement"],
        SymbolType::Function,
    ),
    (&["package_statement"], SymbolType::Module),
];

const JULIA_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_definition", "short_function_definition"],
        SymbolType::Function,
    ),
    (&["struct_definition"], SymbolType::Struct),
    (&["module_definition"], SymbolType::Module),
    (&["abstract_definition"], SymbolType::TypeAlias),
    (&["macro_definition"], SymbolType::Function),
];

const NIX_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_expression", "function"], SymbolType::Function),
    (&["binding", "attrset_expression"], SymbolType::Variable),
    (&["let_expression"], SymbolType::Block),
];

const OCAML_KINDS: &[(&[&str], SymbolType)] = &[
    (&["value_definition", "let_binding"], SymbolType::Function),
    (&["type_definition"], SymbolType::TypeAlias),
    (&["module_definition"], SymbolType::Module),
    (&["module_type_definition"], SymbolType::Interface),
    (&["class_definition"], SymbolType::Class),
    (&["external"], SymbolType::Function),
];

const GROOVY_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_definition", "method_declaration"],
        SymbolType::Function,
    ),
    (
        &["class_definition", "class_declaration"],
        SymbolType::Class,
    ),
    (
        &["interface_definition", "interface_declaration"],
        SymbolType::Interface,
    ),
];

const CLOJURE_KINDS: &[(&[&str], SymbolType)] = &[
    // "list_lit" intentionally unmapped: handled specially — (defn ...) etc
    (&["defn"], SymbolType::Function),
    (&["ns"], SymbolType::Module),
];

const COMMONLISP_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["defun", "defmacro", "defgeneric", "defmethod"],
        SymbolType::Function,
    ),
    (&["defclass"], SymbolType::Class),
    (
        &["defvar", "defparameter", "defconstant"],
        SymbolType::Variable,
    ),
    (&["defpackage"], SymbolType::Module),
    // "list_lit" intentionally unmapped: handled via recursion
];

const ERLANG_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_clause", "fun_expr"], SymbolType::Function),
    (
        &["type_declaration", "record_declaration"],
        SymbolType::TypeAlias,
    ),
    (&["module_attribute"], SymbolType::Module),
];

const FSHARP_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_or_value_defn", "value_declaration"],
        SymbolType::Function,
    ),
    (
        &["type_definition", "type_abbrev_defn"],
        SymbolType::TypeAlias,
    ),
    (&["module_defn"], SymbolType::Module),
    (&["class_defn"], SymbolType::Class),
];

const FORTRAN_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function", "function_statement", "function_subprogram"],
        SymbolType::Function,
    ),
    (
        &[
            "subroutine",
            "subroutine_statement",
            "subroutine_subprogram",
        ],
        SymbolType::Function,
    ),
    (&["module"], SymbolType::Module),
    (
        &["derived_type_definition", "type_statement"],
        SymbolType::Struct,
    ),
    (&["program"], SymbolType::Block),
    // "module_statement" and "program_statement" intentionally unmapped: they
    // are the name-bearing header lines of the enclosing `module`/`program`,
    // which is itself recorded, so mapping them yields a duplicate symbol
    // spanning only the header.
];

const POWERSHELL_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_statement"], SymbolType::Function),
    (&["class_statement"], SymbolType::Class),
    (&["class_method_definition"], SymbolType::Method),
    (&["enum_statement"], SymbolType::Enum),
];

const R_KINDS: &[(&[&str], SymbolType)] = &[(&["function_definition"], SymbolType::Function)];

const MATLAB_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_definition"], SymbolType::Class),
];

const DLANG_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_declaration", "auto_declaration"],
        SymbolType::Function,
    ),
    (&["class_declaration"], SymbolType::Class),
    (&["struct_declaration"], SymbolType::Struct),
    (&["enum_declaration"], SymbolType::Enum),
    (&["interface_declaration"], SymbolType::Interface),
    (&["module_declaration"], SymbolType::Module),
    (&["template_declaration"], SymbolType::TypeAlias),
];

const FISH_KINDS: &[(&[&str], SymbolType)] = &[(&["function_definition"], SymbolType::Function)];

const ZSH_KINDS: &[(&[&str], SymbolType)] = &[(&["function_definition"], SymbolType::Function)];

const LUAU_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["function_declaration", "local_function"],
        SymbolType::Function,
    ),
    (&["type_definition"], SymbolType::TypeAlias),
];

const SCHEME_KINDS: &[(&[&str], SymbolType)] = &[
    (&["define"], SymbolType::Function),
    (&["lambda"], SymbolType::Function),
];

const RACKET_KINDS: &[(&[&str], SymbolType)] = &[
    (&["define"], SymbolType::Function),
    (&["lambda"], SymbolType::Function),
    (&["module"], SymbolType::Module),
    (&["struct"], SymbolType::Struct),
];

const ELM_KINDS: &[(&[&str], SymbolType)] = &[
    (&["value_declaration"], SymbolType::Function),
    (&["type_alias_declaration"], SymbolType::TypeAlias),
    (&["type_declaration"], SymbolType::Enum),
    (&["module_declaration"], SymbolType::Module),
];

const GLSL_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["struct_specifier"], SymbolType::Struct),
    (&["declaration"], SymbolType::Variable),
];

const HLSL_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_specifier"], SymbolType::Class),
    (&["struct_specifier"], SymbolType::Struct),
    (&["declaration"], SymbolType::Variable),
];

const SVELTE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["script_element"], SymbolType::Block),
    (&["style_element"], SymbolType::Block),
    (&["element"], SymbolType::Block),
    (
        &["if_statement", "each_statement", "await_statement"],
        SymbolType::Block,
    ),
];

const ASTRO_KINDS: &[(&[&str], SymbolType)] = &[
    (&["frontmatter"], SymbolType::Block),
    (
        &["element", "script_element", "style_element", "component"],
        SymbolType::Block,
    ),
];

const MAKEFILE_KINDS: &[(&[&str], SymbolType)] = &[
    (&["rule"], SymbolType::Function),
    (&["variable_assignment"], SymbolType::Variable),
    (&["define_directive"], SymbolType::Function),
    (&["include_directive"], SymbolType::Variable),
];

const INI_KINDS: &[(&[&str], SymbolType)] = &[
    (&["section"], SymbolType::Block),
    (&["setting"], SymbolType::Variable),
];

const NGINX_KINDS: &[(&[&str], SymbolType)] = &[
    (&["block"], SymbolType::Block),
    (&["directive"], SymbolType::Variable),
];

const PRISMA_KINDS: &[(&[&str], SymbolType)] = &[
    (&["model_declaration"], SymbolType::Struct),
    (&["enum_declaration"], SymbolType::Enum),
    (
        &["generator_declaration", "datasource_declaration"],
        SymbolType::Block,
    ),
    (&["type_declaration"], SymbolType::TypeAlias),
];

const RUST_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_item"], SymbolType::Function),
    (&["impl_item"], SymbolType::Block),
    (&["struct_item"], SymbolType::Struct),
    (&["enum_item"], SymbolType::Enum),
    (&["trait_item"], SymbolType::Trait),
    (&["type_item"], SymbolType::TypeAlias),
    (&["const_item"], SymbolType::Constant),
    (&["static_item"], SymbolType::Constant),
    (&["mod_item"], SymbolType::Module),
];

const TYPESCRIPT_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_declaration"], SymbolType::Function),
    (&["class_declaration"], SymbolType::Class),
    (&["interface_declaration"], SymbolType::Interface),
    (&["type_alias_declaration"], SymbolType::TypeAlias),
    (&["enum_declaration"], SymbolType::Enum),
    // `method_signature` covers TS interface methods and abstract class
    // members (`abstract_method_signature`): body-less declarations that
    // still name a callable member.
    (
        &[
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        ],
        SymbolType::Method,
    ),
    (
        &["lexical_declaration", "variable_declaration"],
        SymbolType::Variable,
    ),
    // "export_statement" intentionally unmapped: recurse into children
];

const PYTHON_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_definition"], SymbolType::Class),
    // "decorated_definition" intentionally unmapped: recurse into children
];

const GO_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_declaration"], SymbolType::Function),
    (&["method_declaration"], SymbolType::Method),
    // "type_declaration" intentionally unmapped: contains type_spec children
    (&["type_spec"], SymbolType::TypeAlias), // refined by child kind
];

const JAVA_KINDS: &[(&[&str], SymbolType)] = &[
    (&["method_declaration"], SymbolType::Method),
    (&["class_declaration"], SymbolType::Class),
    (&["interface_declaration"], SymbolType::Interface),
    (&["enum_declaration"], SymbolType::Enum),
    (&["constructor_declaration"], SymbolType::Method),
];

const C_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["struct_specifier"], SymbolType::Struct),
    (&["enum_specifier"], SymbolType::Enum),
    (&["type_definition"], SymbolType::TypeAlias),
    (&["declaration"], SymbolType::Variable),
];

const CPP_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_specifier"], SymbolType::Class),
    (&["struct_specifier"], SymbolType::Struct),
    (&["enum_specifier"], SymbolType::Enum),
    (&["type_definition"], SymbolType::TypeAlias),
    (&["namespace_definition"], SymbolType::Module),
    // "template_declaration" intentionally unmapped: recurse into children
    (&["declaration"], SymbolType::Variable),
];

const RUBY_KINDS: &[(&[&str], SymbolType)] = &[
    (&["method", "singleton_method"], SymbolType::Function),
    (&["class"], SymbolType::Class),
    (&["module"], SymbolType::Module),
];

const BASH_KINDS: &[(&[&str], SymbolType)] = &[(&["function_definition"], SymbolType::Function)];

const KOTLIN_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_declaration"], SymbolType::Function),
    (&["class_declaration"], SymbolType::Class), // refined later
    (&["object_declaration"], SymbolType::Class), // objects treated as classes/singletons
];

const SWIFT_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_declaration"], SymbolType::Function),
    (&["class_declaration"], SymbolType::Class), // refined later
    (&["protocol_declaration"], SymbolType::Interface),
];

const ZIG_KINDS: &[(&[&str], SymbolType)] = &[(&["function_declaration"], SymbolType::Function)];

const LUA_KINDS: &[(&[&str], SymbolType)] = &[(&["function_declaration"], SymbolType::Function)];

const SCALA_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_definition"], SymbolType::Class),
    (&["trait_definition"], SymbolType::Trait),
    (&["object_definition"], SymbolType::Module), // objects map well to modules in scala
];

const CSHARP_KINDS: &[(&[&str], SymbolType)] = &[
    (&["class_declaration"], SymbolType::Class),
    (&["interface_declaration"], SymbolType::Interface),
    (&["struct_declaration"], SymbolType::Struct),
    (&["enum_declaration"], SymbolType::Enum),
    (
        &["method_declaration", "local_function_statement"],
        SymbolType::Method,
    ),
    (
        &["namespace_declaration", "file_scoped_namespace_declaration"],
        SymbolType::Module,
    ),
];

const PHP_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function_definition"], SymbolType::Function),
    (&["class_declaration"], SymbolType::Class),
    (&["interface_declaration"], SymbolType::Interface),
    (&["method_declaration"], SymbolType::Method),
];

const HASKELL_KINDS: &[(&[&str], SymbolType)] = &[
    (&["function", "signature"], SymbolType::Function),
    (&["data_type"], SymbolType::Struct),
    (&["type_alias", "type_synomym"], SymbolType::TypeAlias),
    (&["newtype"], SymbolType::TypeAlias),
];

const DART_KINDS: &[(&[&str], SymbolType)] = &[
    (
        &["class_declaration", "class_definition"],
        SymbolType::Class,
    ),
    (&["enum_declaration"], SymbolType::Enum),
    (
        &["function_signature", "function_definition"],
        SymbolType::Function,
    ),
    (
        &["method_signature", "method_definition"],
        SymbolType::Method,
    ),
];
