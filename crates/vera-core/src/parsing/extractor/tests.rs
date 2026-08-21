use super::*;
use crate::parsing::languages::tree_sitter_grammar;

fn parse_and_extract(source: &str, lang: Language) -> Vec<RawSymbol> {
    let grammar = tree_sitter_grammar(lang).unwrap();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&grammar).unwrap();
    let tree = parser.parse(source, None).unwrap();
    extract_symbols(&tree, source.as_bytes(), lang)
}

#[test]
fn rust_extracts_functions() {
    let source = r#"
fn hello() {
    println!("hello");
}

fn world(x: i32) -> i32 {
    x + 1
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    assert_eq!(symbols.len(), 2);
    assert_eq!(symbols[0].name.as_deref(), Some("hello"));
    assert_eq!(symbols[0].symbol_type, SymbolType::Function);
    assert_eq!(symbols[1].name.as_deref(), Some("world"));
}

#[test]
fn rust_extracts_structs_and_enums() {
    let source = r#"
struct Point {
    x: f64,
    y: f64,
}

enum Color {
    Red,
    Green,
    Blue,
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    assert_eq!(symbols.len(), 2);
    assert_eq!(symbols[0].symbol_type, SymbolType::Struct);
    assert_eq!(symbols[0].name.as_deref(), Some("Point"));
    assert_eq!(symbols[1].symbol_type, SymbolType::Enum);
    assert_eq!(symbols[1].name.as_deref(), Some("Color"));
}

#[test]
fn rust_extracts_impl_methods() {
    let source = r#"
impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
        }
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    assert_eq!(symbols.len(), 3);
    assert_eq!(symbols[0].symbol_type, SymbolType::Block);
    assert_eq!(symbols[0].name.as_deref(), Some("impl Point"));
    assert_eq!(symbols[1].symbol_type, SymbolType::Method);
    assert_eq!(symbols[1].name.as_deref(), Some("new"));
    assert_eq!(symbols[2].symbol_type, SymbolType::Method);
    assert_eq!(symbols[2].name.as_deref(), Some("distance"));
}

#[test]
fn rust_extracts_trait_impl_names() {
    let source = r#"
impl Sink for StdoutSink {
    fn write(&self) {}
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    assert_eq!(symbols[0].symbol_type, SymbolType::Block);
    assert_eq!(symbols[0].name.as_deref(), Some("impl Sink for StdoutSink"));
}

#[test]
fn rust_extracts_traits() {
    let source = r#"
trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].symbol_type, SymbolType::Trait);
    assert_eq!(symbols[0].name.as_deref(), Some("Drawable"));
}

/// A `#[cfg(test)] mod tests { ... }` block holding the shapes that used to be
/// swallowed by the module symbol: a free function, a struct, and an impl.
const INLINE_MOD_SOURCE: &str = r#"
fn helper() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> i32 {
        helper()
    }

    struct Case {
        value: i32,
    }

    impl Case {
        fn run(&self) -> i32 {
            fixture()
        }
    }
}
"#;

#[test]
fn rust_inline_mod_does_not_swallow_inner_symbols() {
    let symbols = parse_and_extract(INLINE_MOD_SOURCE, Language::Rust);
    let named: Vec<(&str, SymbolType)> = symbols
        .iter()
        .filter_map(|s| s.name.as_deref().map(|n| (n, s.symbol_type)))
        .collect();

    // Presence first: an extractor that returned nothing must not pass this.
    assert!(
        named.contains(&("helper", SymbolType::Function)),
        "expected the top-level function, got {named:?}"
    );
    assert!(
        named.contains(&("fixture", SymbolType::Function)),
        "expected the function inside the inline mod, got {named:?}"
    );
    assert!(
        named.contains(&("Case", SymbolType::Struct)),
        "expected the struct inside the inline mod, got {named:?}"
    );
    assert!(
        named.contains(&("run", SymbolType::Method)),
        "expected the impl method inside the inline mod, got {named:?}"
    );
    // The module itself stays indexable, as `impl` blocks do.
    assert!(
        named.contains(&("tests", SymbolType::Module)),
        "expected the module symbol itself, got {named:?}"
    );

    // Only then: the module no longer spans the inner symbols on its own.
    let module = symbols
        .iter()
        .find(|s| s.symbol_type == SymbolType::Module)
        .expect("module symbol");
    let fixture = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("fixture"))
        .expect("fixture symbol");
    assert!(
        fixture.start_byte > module.start_byte && fixture.end_byte < module.end_byte,
        "fixture should be nested strictly inside the module span"
    );
}

#[test]
fn rust_inline_mod_attributes_calls_to_the_calling_function() {
    let grammar = tree_sitter_grammar(Language::Rust).unwrap();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&grammar).unwrap();
    let tree = parser.parse(INLINE_MOD_SOURCE, None).unwrap();
    let refs = crate::parsing::references::extract_references(
        &tree,
        INLINE_MOD_SOURCE.as_bytes(),
        Language::Rust,
    );

    let callers = |callee: &str| -> Vec<Option<String>> {
        refs.iter()
            .filter(|r| r.callee == callee)
            .map(|r| r.caller.clone())
            .collect()
    };

    // Presence first: the calls have to be seen at all.
    assert_eq!(
        callers("helper"),
        vec![Some("fixture".to_string())],
        "call to helper should be attributed to the calling function, not the module"
    );
    assert_eq!(
        callers("fixture"),
        vec![Some("run".to_string())],
        "call to fixture should be attributed to the calling method, not the module"
    );
}

#[test]
fn rust_trait_extracts_default_methods() {
    let source = r#"
trait Greeter {
    fn name(&self) -> String;

    fn greet(&self) -> String {
        format!("hi {}", self.name())
    }
}
"#;
    let symbols = parse_and_extract(source, Language::Rust);
    let named: Vec<(&str, SymbolType)> = symbols
        .iter()
        .filter_map(|s| s.name.as_deref().map(|n| (n, s.symbol_type)))
        .collect();
    assert!(
        named.contains(&("Greeter", SymbolType::Trait)),
        "expected the trait symbol, got {named:?}"
    );
    assert!(
        named.contains(&("greet", SymbolType::Method)),
        "expected the default method body, got {named:?}"
    );
}

#[test]
fn python_extracts_functions_and_class_methods() {
    let source = r#"
def hello():
    print("hello")

class MyClass:
    def __init__(self):
        self.x = 0

    def method(self):
        return self.x
"#;
    let symbols = parse_and_extract(source, Language::Python);
    // Should extract: hello (function), MyClass (class), __init__ (method), method (method)
    assert_eq!(symbols.len(), 4);
    assert_eq!(symbols[0].symbol_type, SymbolType::Function);
    assert_eq!(symbols[0].name.as_deref(), Some("hello"));
    assert_eq!(symbols[1].symbol_type, SymbolType::Class);
    assert_eq!(symbols[1].name.as_deref(), Some("MyClass"));
    assert_eq!(symbols[2].symbol_type, SymbolType::Method);
    assert_eq!(symbols[2].name.as_deref(), Some("__init__"));
    assert_eq!(symbols[3].symbol_type, SymbolType::Method);
    assert_eq!(symbols[3].name.as_deref(), Some("method"));
}

#[test]
fn python_class_without_methods_kept_as_class() {
    let source = r#"
class Config:
    DEBUG = True
    VERSION = "1.0"
"#;
    let symbols = parse_and_extract(source, Language::Python);
    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].symbol_type, SymbolType::Class);
    assert_eq!(symbols[0].name.as_deref(), Some("Config"));
}

#[test]
fn python_decorated_methods_extracted_separately() {
    let source = r#"
class MyClass:
    @staticmethod
    def static_method():
        pass

    @property
    def value(self):
        return self._value
"#;
    let symbols = parse_and_extract(source, Language::Python);
    assert_eq!(symbols.len(), 3);
    assert_eq!(symbols[0].symbol_type, SymbolType::Class);
    assert_eq!(symbols[0].name.as_deref(), Some("MyClass"));
    assert_eq!(symbols[1].symbol_type, SymbolType::Method);
    assert_eq!(symbols[1].name.as_deref(), Some("static_method"));
    assert_eq!(symbols[2].symbol_type, SymbolType::Method);
    assert_eq!(symbols[2].name.as_deref(), Some("value"));
}

#[test]
fn typescript_extracts_functions_and_interfaces() {
    let source = r#"
function greet(name: string): string {
    return `Hello, ${name}!`;
}

interface User {
    name: string;
    age: number;
}

class UserService {
    private users: User[];

    getUser(id: number): User {
        return this.users[id];
    }
}
"#;
    let symbols = parse_and_extract(source, Language::TypeScript);
    assert!(
        symbols.len() >= 4,
        "expected >= 4 symbols, got {}",
        symbols.len()
    );

    let func = symbols.iter().find(|s| s.name.as_deref() == Some("greet"));
    assert!(func.is_some(), "should find function 'greet'");
    assert_eq!(func.unwrap().symbol_type, SymbolType::Function);

    let iface = symbols.iter().find(|s| s.name.as_deref() == Some("User"));
    assert!(iface.is_some(), "should find interface 'User'");
    assert_eq!(iface.unwrap().symbol_type, SymbolType::Interface);

    let class = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("UserService"));
    assert!(class.is_some(), "should find class 'UserService'");
    assert_eq!(class.unwrap().symbol_type, SymbolType::Class);

    let method = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("getUser"));
    assert!(method.is_some(), "should find method 'getUser'");
    assert_eq!(method.unwrap().symbol_type, SymbolType::Method);
}

#[test]
fn go_extracts_functions_and_structs() {
    let source = r#"
package main

func Hello() string {
    return "hello"
}

type Point struct {
    X float64
    Y float64
}

func (p *Point) Distance() float64 {
    return p.X * p.X + p.Y * p.Y
}
"#;
    let symbols = parse_and_extract(source, Language::Go);
    assert!(
        symbols.len() >= 3,
        "expected >= 3 symbols, got {}",
        symbols.len()
    );

    let func = symbols.iter().find(|s| s.name.as_deref() == Some("Hello"));
    assert!(func.is_some(), "should find function 'Hello'");
    assert_eq!(func.unwrap().symbol_type, SymbolType::Function);

    let struc = symbols.iter().find(|s| s.name.as_deref() == Some("Point"));
    assert!(struc.is_some(), "should find struct 'Point'");
    assert_eq!(struc.unwrap().symbol_type, SymbolType::Struct);

    let method = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Distance"));
    assert!(method.is_some(), "should find method 'Distance'");
    assert_eq!(method.unwrap().symbol_type, SymbolType::Method);
}

#[test]
fn java_extracts_class_and_methods() {
    let source = r#"
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"#;
    let symbols = parse_and_extract(source, Language::Java);
    assert!(
        symbols.len() >= 3,
        "expected >= 3 symbols, got {}",
        symbols.len()
    );

    let class = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Calculator"));
    assert!(class.is_some(), "should find class 'Calculator'");
    assert_eq!(class.unwrap().symbol_type, SymbolType::Class);

    let add = symbols.iter().find(|s| s.name.as_deref() == Some("add"));
    assert!(add.is_some(), "should find method 'add'");
    assert_eq!(add.unwrap().symbol_type, SymbolType::Method);

    let multiply = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("multiply"));
    assert!(multiply.is_some(), "should find method 'multiply'");
    assert_eq!(multiply.unwrap().symbol_type, SymbolType::Method);
}

#[test]
fn java_enum_extracts_methods() {
    let source = r#"
enum Direction {
    UP,
    DOWN;

    int code() {
        return ordinal();
    }
}
"#;
    let symbols = parse_and_extract(source, Language::Java);
    assert!(
        symbols.len() >= 2,
        "expected >= 2 symbols, got {}",
        symbols.len()
    );

    let enm = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Direction"));
    assert!(enm.is_some(), "should find enum 'Direction'");
    assert_eq!(enm.unwrap().symbol_type, SymbolType::Enum);

    let method = symbols.iter().find(|s| s.name.as_deref() == Some("code"));
    assert!(method.is_some(), "should find enum method 'code'");
    assert_eq!(method.unwrap().symbol_type, SymbolType::Method);
}

#[test]
fn c_extracts_functions_and_structs() {
    let source = r#"
struct Point {
    double x;
    double y;
};

double distance(struct Point* p) {
    return p->x * p->x + p->y * p->y;
}
"#;
    let symbols = parse_and_extract(source, Language::C);
    assert!(
        !symbols.is_empty(),
        "expected >= 1 symbol, got {}",
        symbols.len()
    );

    let func = symbols
        .iter()
        .find(|s| s.symbol_type == SymbolType::Function);
    assert!(func.is_some(), "should find a function");
    assert_eq!(func.unwrap().name.as_deref(), Some("distance"));
}

#[test]
fn cpp_extracts_classes_and_functions() {
    let source = r#"
class Shape {
public:
    virtual double area() = 0;
};

namespace geometry {
    double pi() {
        return 3.14159;
    }
}
"#;
    let symbols = parse_and_extract(source, Language::Cpp);
    assert!(
        !symbols.is_empty(),
        "expected >= 1 symbol, got {}",
        symbols.len()
    );

    let class = symbols.iter().find(|s| s.symbol_type == SymbolType::Class);
    assert!(class.is_some(), "should find a class");

    let ns = symbols.iter().find(|s| s.symbol_type == SymbolType::Module);
    assert!(ns.is_some(), "should find a namespace");
}

#[test]
fn ruby_extracts_functions_and_classes() {
    let source = r#"
module MyModule
end

class MyClass
end

def my_method
end
"#;
    let symbols = parse_and_extract(source, Language::Ruby);
    assert!(symbols.len() >= 3);
    let m = symbols.iter().find(|s| s.symbol_type == SymbolType::Module);
    assert!(m.is_some());
    assert_eq!(m.unwrap().name.as_deref(), Some("MyModule"));

    let c = symbols.iter().find(|s| s.symbol_type == SymbolType::Class);
    assert!(c.is_some());
    assert_eq!(c.unwrap().name.as_deref(), Some("MyClass"));

    let f = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("my_method"));
    assert!(f.is_some());
    assert_eq!(f.unwrap().symbol_type, SymbolType::Function);
}

#[test]
fn bash_extracts_functions() {
    let source = r#"
function foo() {
    echo "foo"
}
bar() {
    echo "bar"
}
"#;
    let symbols = parse_and_extract(source, Language::Bash);
    assert_eq!(symbols.len(), 2);
    assert_eq!(symbols[0].symbol_type, SymbolType::Function);
    assert_eq!(symbols[0].name.as_deref(), Some("foo"));
    assert_eq!(symbols[1].symbol_type, SymbolType::Function);
    assert_eq!(symbols[1].name.as_deref(), Some("bar"));
}

#[test]
fn kotlin_extracts_types_and_functions() {
    let source = r#"
fun foo() {}
class Bar {}
interface Baz {}
object Qux {}
enum class Quux {}
"#;
    let symbols = parse_and_extract(source, Language::Kotlin);
    assert_eq!(symbols.len(), 5);

    let fun = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("foo"))
        .unwrap();
    assert_eq!(fun.symbol_type, SymbolType::Function);

    let cls = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Bar"))
        .unwrap();
    assert_eq!(cls.symbol_type, SymbolType::Class);

    let iface = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Baz"))
        .unwrap();
    assert_eq!(iface.symbol_type, SymbolType::Interface);

    let obj = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Qux"))
        .unwrap();
    assert_eq!(obj.symbol_type, SymbolType::Class);

    let enm = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Quux"))
        .unwrap();
    assert_eq!(enm.symbol_type, SymbolType::Enum);
}

#[test]
fn swift_extracts_types_and_functions() {
    let source = r#"
func foo() {}
class Bar {}
struct Baz {}
enum Qux {}
protocol Quux {}
"#;
    let symbols = parse_and_extract(source, Language::Swift);
    assert_eq!(symbols.len(), 5);

    let fun = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("foo"))
        .unwrap();
    assert_eq!(fun.symbol_type, SymbolType::Function);

    let cls = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Bar"))
        .unwrap();
    assert_eq!(cls.symbol_type, SymbolType::Class);

    let strc = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Baz"))
        .unwrap();
    assert_eq!(strc.symbol_type, SymbolType::Struct);

    let enm = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Qux"))
        .unwrap();
    assert_eq!(enm.symbol_type, SymbolType::Enum);

    let proto = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Quux"))
        .unwrap();
    assert_eq!(proto.symbol_type, SymbolType::Interface);
}

#[test]
fn zig_extracts_functions_and_structs() {
    let source = r#"
fn foo() void {}
const Bar = struct {};
"#;
    let symbols = parse_and_extract(source, Language::Zig);
    assert_eq!(symbols.len(), 2);

    let fun = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("foo"))
        .unwrap();
    assert_eq!(fun.symbol_type, SymbolType::Function);

    let strc = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("Bar"))
        .unwrap();
    assert_eq!(strc.symbol_type, SymbolType::Struct);
}

#[test]
fn lua_extracts_functions() {
    let source = r#"
function foo() end
local function bar() end
"#;
    let symbols = parse_and_extract(source, Language::Lua);
    assert_eq!(symbols.len(), 2);

    let foo = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("foo"))
        .unwrap();
    assert_eq!(foo.symbol_type, SymbolType::Function);

    let bar = symbols
        .iter()
        .find(|s| s.name.as_deref() == Some("bar"))
        .unwrap();
    assert_eq!(bar.symbol_type, SymbolType::Function);
}

#[test]
fn scala_extracts_types_and_functions() {
    let source = r#"
def main(args: Array[String]): Unit = {}
class User {}
trait Logger {}
"#;
    let symbols = parse_and_extract(source, Language::Scala);
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("main"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Class && s.name.as_deref() == Some("User"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Trait && s.name.as_deref() == Some("Logger"))
    );
}

#[test]
fn csharp_extracts_types_and_methods() {
    let source = r#"
namespace MyApp {
    class Calculator {
        public int Add(int a, int b) { return a + b; }
    }
    interface IWorker {}
    struct Point {}
    enum Status { Ok, Error }
}
"#;
    let symbols = parse_and_extract(source, Language::CSharp);
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Module));
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Class));
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Method));
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Interface)
    );
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Struct));
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Enum));
}

#[test]
fn php_extracts_classes_and_functions() {
    let source = r#"
<?php
function foo() {}
class Bar {
    public function baz() {}
}
interface Qux {}
"#;
    let symbols = parse_and_extract(source, Language::Php);
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function)
    );
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Class));
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Method));
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Interface)
    );
}

#[test]
fn haskell_extracts_types_and_functions() {
    let source = r#"
data Point = Point Float Float
type Age = Int
add :: Int -> Int -> Int
add x y = x + y
"#;
    let symbols = parse_and_extract(source, Language::Haskell);
    assert!(symbols.iter().any(|s| s.symbol_type == SymbolType::Struct));
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::TypeAlias)
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function)
    );
}

#[test]
fn elixir_extracts_modules_and_functions() {
    let source = r#"
defmodule Math do
  def add(a, b) do
    a + b
  end
  defp sub(a, b), do: a - b
  defmacro mul(a, b) do
  end
end
"#;
    let symbols = parse_and_extract(source, Language::Elixir);
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Module && s.name.as_deref() == Some("Math"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("add"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("sub"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function && s.name.as_deref() == Some("mul"))
    );
}

#[test]
fn hcl_extracts_blocks() {
    let source = r#"
resource "aws_instance" "web" {
  ami = "ami-123"
}
module "vpc" {}
data "aws_ami" "ubuntu" {}
variable "image_id" {}
output "instance_ip" {}
"#;
    let symbols = parse_and_extract(source, Language::Hcl);
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Struct
                && s.name.as_deref() == Some("aws_instance"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Module && s.name.as_deref() == Some("vpc"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Struct && s.name.as_deref() == Some("aws_ami"))
    );
    assert!(
        symbols.iter().any(
            |s| s.symbol_type == SymbolType::TypeAlias && s.name.as_deref() == Some("image_id")
        )
    );
    assert!(symbols.iter().any(
        |s| s.symbol_type == SymbolType::TypeAlias && s.name.as_deref() == Some("instance_ip")
    ));
}

#[test]
fn sql_extracts_statements() {
    let source = r#"
CREATE TABLE users (id INT);
CREATE FUNCTION get_user() RETURNS INT AS $$ SELECT 1 $$ LANGUAGE SQL;
CREATE VIEW active_users AS SELECT * FROM users;
CREATE PROCEDURE my_proc() LANGUAGE SQL AS $$ $$;
"#;
    let symbols = parse_and_extract(source, Language::Sql);
    assert!(!symbols.is_empty());
}

#[test]
fn protobuf_extracts_types() {
    let source = r#"
syntax = "proto3";
message User { int32 id = 1; }
enum Status { ACTIVE = 0; }
service AuthService { rpc Login(User) returns (User); }
"#;
    let symbols = parse_and_extract(source, Language::Protobuf);
    assert!(!symbols.is_empty());
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Struct && s.name.as_deref() == Some("User"))
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Enum && s.name.as_deref() == Some("Status"))
    );
    assert!(symbols.iter().any(
            |s| s.symbol_type == SymbolType::Class && s.name.as_deref() == Some("AuthService")
        ));
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Method && s.name.as_deref() == Some("Login"))
    );
}

// ── Tier 1B symbol extraction tests ─────────────────────────

#[test]
fn html_extracts_elements() {
    let source = r#"
<html>
<head><title>Test</title></head>
<body>
  <div class="container">
    <p>Hello</p>
  </div>
  <script>console.log("hi")</script>
  <style>body { color: red; }</style>
</body>
</html>
"#;
    let symbols = parse_and_extract(source, Language::Html);
    assert!(!symbols.is_empty(), "HTML should extract some symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "HTML should extract block symbols"
    );
}

#[test]
fn css_extracts_rules() {
    let source = r#"
body {
    color: red;
    font-size: 16px;
}

.container {
    max-width: 1200px;
}

@media (max-width: 768px) {
    body { font-size: 14px; }
}
"#;
    let symbols = parse_and_extract(source, Language::Css);
    assert!(!symbols.is_empty(), "CSS should extract some symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "CSS should extract rule_set blocks"
    );
}

#[test]
fn scss_extracts_mixins_and_rules() {
    let source = r#"
$primary: #333;

@mixin flex-center {
    display: flex;
    align-items: center;
}

.container {
    @include flex-center;
    color: $primary;
}
"#;
    let symbols = parse_and_extract(source, Language::Scss);
    assert!(!symbols.is_empty(), "SCSS should extract some symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "SCSS should extract mixin as function"
    );
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "SCSS should extract rule_set blocks"
    );
}

#[test]
fn vue_extracts_sections() {
    let source = r#"
<template>
  <div>{{ message }}</div>
</template>

<script>
export default {
  data() { return { message: "Hello" } }
}
</script>

<style>
.container { color: red; }
</style>
"#;
    let symbols = parse_and_extract(source, Language::Vue);
    assert!(!symbols.is_empty(), "Vue should extract some symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "Vue should extract template/script/style blocks"
    );
}

#[test]
fn graphql_extracts_types_and_queries() {
    let source = r#"
type User {
    id: ID!
    name: String!
    email: String
}

enum Role {
    ADMIN
    USER
    GUEST
}

interface Node {
    id: ID!
}

input CreateUserInput {
    name: String!
    email: String!
}

query GetUser($id: ID!) {
    user(id: $id) {
        name
        email
    }
}
"#;
    let symbols = parse_and_extract(source, Language::GraphQl);
    assert!(!symbols.is_empty(), "GraphQL should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
        "GraphQL should extract type as struct"
    );
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Enum),
        "GraphQL should extract enum"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Interface),
        "GraphQL should extract interface"
    );
}

#[test]
fn cmake_extracts_functions_and_commands() {
    let source = r#"
cmake_minimum_required(VERSION 3.10)
project(MyProject)

function(my_helper ARG)
    message(STATUS "${ARG}")
endfunction()

macro(my_macro)
    set(MY_VAR "value")
endmacro()

add_executable(main src/main.cpp)
"#;
    let symbols = parse_and_extract(source, Language::CMake);
    assert!(!symbols.is_empty(), "CMake should extract symbols");
}

#[test]
fn dockerfile_extracts_instructions() {
    let source = r#"FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_VERSION=1.0

RUN apt-get update && apt-get install -y curl

COPY . /app
WORKDIR /app

CMD ["./app"]
"#;
    let symbols = parse_and_extract(source, Language::Dockerfile);
    assert!(!symbols.is_empty(), "Dockerfile should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "Dockerfile should extract FROM as block"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Constant),
        "Dockerfile should extract ENV/ARG as constant"
    );
}

#[test]
fn xml_extracts_elements() {
    let source = r#"<?xml version="1.0" encoding="UTF-8"?>
<project>
    <name>MyProject</name>
    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>lib</artifactId>
        </dependency>
    </dependencies>
</project>
"#;
    let symbols = parse_and_extract(source, Language::Xml);
    assert!(!symbols.is_empty(), "XML should extract some symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "XML should extract elements as blocks"
    );
}

// ── Tier 2A symbol extraction tests ─────────────────────────

#[test]
fn objectivec_extracts_classes_and_methods() {
    let source = r#"
@interface Calculator : NSObject
- (int)add:(int)a to:(int)b;
@end

@implementation Calculator
- (int)add:(int)a to:(int)b {
    return a + b;
}
@end
"#;
    let symbols = parse_and_extract(source, Language::ObjectiveC);
    assert!(!symbols.is_empty(), "ObjC should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Class),
        "ObjC should extract class interface/implementation"
    );
    // Methods are extracted inside class body
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Method),
        "ObjC should extract methods"
    );
}

#[test]
fn perl_extracts_functions() {
    let source = r#"
package MyModule;

sub hello {
    print "hello\n";
}

sub world {
    my ($name) = @_;
    print "hello $name\n";
}

1;
"#;
    let symbols = parse_and_extract(source, Language::Perl);
    assert!(!symbols.is_empty(), "Perl should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Perl should extract functions"
    );
}

#[test]
fn julia_extracts_functions_and_structs() {
    let source = r#"
function hello()
    println("hello")
end

struct Point
    x::Float64
    y::Float64
end

module MyModule
end
"#;
    let symbols = parse_and_extract(source, Language::Julia);
    assert!(!symbols.is_empty(), "Julia should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Julia should extract functions"
    );
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
        "Julia should extract structs"
    );
}

#[test]
fn nix_extracts_bindings() {
    let source = r#"
{
  hello = "world";
  foo = x: x + 1;
}
"#;
    let symbols = parse_and_extract(source, Language::Nix);
    assert!(!symbols.is_empty(), "Nix should extract symbols");
}

#[test]
fn ocaml_extracts_functions_and_types() {
    let source = r#"
let hello () = print_endline "hello"

type point = { x: float; y: float }

module MyModule = struct end
"#;
    let symbols = parse_and_extract(source, Language::OCaml);
    assert!(!symbols.is_empty(), "OCaml should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "OCaml should extract functions"
    );
}

#[test]
fn groovy_extracts_classes_and_functions() {
    let source = r#"
class Calculator {
    int add(int a, int b) {
        return a + b
    }
}

def hello() {
    println "hello"
}
"#;
    let symbols = parse_and_extract(source, Language::Groovy);
    assert!(!symbols.is_empty(), "Groovy should extract symbols");
}

#[test]
fn clojure_extracts_functions() {
    let source = r#"
(ns myapp.core)

(defn hello []
  (println "hello"))

(defn add [a b]
  (+ a b))
"#;
    let symbols = parse_and_extract(source, Language::Clojure);
    assert!(
        symbols.len() >= 3,
        "Clojure should extract at least 3 symbols (ns + 2 defn), got {}",
        symbols.len()
    );
    assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("myapp.core")
                    && s.symbol_type == SymbolType::Module),
            "Clojure should extract ns 'myapp.core'"
        );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function),
        "Clojure should extract function 'hello'"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
        "Clojure should extract function 'add'"
    );
}

#[test]
fn clojure_extracts_defmacro() {
    let source = "(defmacro my-when [test & body]\n  `(if ~test (do ~@body)))\n";
    let symbols = parse_and_extract(source, Language::Clojure);
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("my-when") && s.symbol_type == SymbolType::Function),
        "Clojure should extract defmacro 'my-when'"
    );
}

#[test]
fn clojure_extracts_def() {
    let source = "(def pi 3.14159)\n";
    let symbols = parse_and_extract(source, Language::Clojure);
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("pi") && s.symbol_type == SymbolType::Variable),
        "Clojure should extract def 'pi'"
    );
}

#[test]
fn commonlisp_extracts_functions() {
    let source = r#"
(defun hello ()
  (format t "hello~%"))

(defun add (a b)
  (+ a b))

(defclass point ()
  ((x :initarg :x)
   (y :initarg :y)))
"#;
    let symbols = parse_and_extract(source, Language::CommonLisp);
    assert!(
        symbols.len() >= 3,
        "CL should extract at least 3 symbols (2 defun + defclass), got {}",
        symbols.len()
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function),
        "CL should extract function 'hello'"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
        "CL should extract function 'add'"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("point") && s.symbol_type == SymbolType::Class),
        "CL should extract class 'point'"
    );
}

#[test]
fn commonlisp_extracts_defvar() {
    let source = "(defvar *my-var* 42)\n";
    let symbols = parse_and_extract(source, Language::CommonLisp);
    assert!(
            symbols
                .iter()
                .any(|s| s.name.as_deref() == Some("*my-var*")
                    && s.symbol_type == SymbolType::Variable),
            "CL should extract defvar '*my-var*'"
        );
}

#[test]
fn erlang_extracts_functions() {
    let source = r#"
-module(hello).
-export([hello/0, add/2]).

hello() -> ok.

add(A, B) -> A + B.
"#;
    let symbols = parse_and_extract(source, Language::Erlang);
    assert!(!symbols.is_empty(), "Erlang should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Erlang should extract functions"
    );
}

#[test]
fn fsharp_extracts_functions() {
    let source = r#"
let hello () = printfn "hello"

let add a b = a + b

type Point = { X: float; Y: float }
"#;
    let symbols = parse_and_extract(source, Language::FSharp);
    assert!(!symbols.is_empty(), "F# should extract symbols");
}

#[test]
fn fortran_extracts_functions_and_subroutines() {
    let source = r#"
program hello
  print *, 'hello'
end program hello

subroutine greet(name)
  character(*), intent(in) :: name
  print *, 'Hello ', name
end subroutine greet

function add(a, b) result(c)
  integer, intent(in) :: a, b
  integer :: c
  c = a + b
end function add
"#;
    let symbols = parse_and_extract(source, Language::Fortran);
    assert!(!symbols.is_empty(), "Fortran should extract symbols");
}

#[test]
fn powershell_extracts_functions_and_classes() {
    let source = r#"
function Get-Hello {
    Write-Host "hello"
}

function Add-Numbers {
    param($a, $b)
    return $a + $b
}

class Calculator {
    [int] Add([int]$a, [int]$b) {
        return $a + $b
    }
}
"#;
    let symbols = parse_and_extract(source, Language::PowerShell);
    assert!(!symbols.is_empty(), "PowerShell should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "PowerShell should extract functions"
    );
}

#[test]
fn r_extracts_functions() {
    let source = r#"
hello <- function() {
  print("hello")
}

add <- function(a, b) {
  a + b
}
"#;
    let symbols = parse_and_extract(source, Language::R);
    assert!(!symbols.is_empty(), "R should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "R should extract functions"
    );
    assert!(
        symbols.iter().any(|s| s.name.as_deref() == Some("hello")),
        "R should extract function name 'hello'"
    );
}

// ── Tier 2A batch 2 symbol extraction tests ─────────────────

#[test]
fn matlab_extracts_functions() {
    let source = r#"
function y = square(x)
  y = x^2;
end

function result = add(a, b)
  result = a + b;
end
"#;
    let symbols = parse_and_extract(source, Language::Matlab);
    assert!(!symbols.is_empty(), "MATLAB should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "MATLAB should extract functions"
    );
}

#[test]
fn dlang_extracts_functions_and_classes() {
    let source = r#"
void main() {
    writeln("hello");
}

class Foo {
    int bar() { return 42; }
}

struct Point {
    float x, y;
}
"#;
    let symbols = parse_and_extract(source, Language::DLang);
    assert!(!symbols.is_empty(), "D should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "D should extract functions"
    );
}

#[test]
fn fish_extracts_functions() {
    let source = r#"
function hello
  echo hello
end

function greet -a name
  echo "Hello, $name"
end
"#;
    let symbols = parse_and_extract(source, Language::Fish);
    assert!(!symbols.is_empty(), "Fish should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Fish should extract functions"
    );
}

#[test]
fn zsh_extracts_functions() {
    let source = r#"
function hello() {
  echo hello
}

greet() {
  echo "Hello, $1"
}
"#;
    let symbols = parse_and_extract(source, Language::Zsh);
    assert!(!symbols.is_empty(), "Zsh should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Zsh should extract functions"
    );
}

#[test]
fn luau_extracts_functions() {
    let source = r#"
local function hello()
  print("hello")
end

function greet(name: string)
  print("Hello, " .. name)
end
"#;
    let symbols = parse_and_extract(source, Language::Luau);
    assert!(!symbols.is_empty(), "Luau should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Luau should extract functions"
    );
}

#[test]
fn scheme_extracts_definitions() {
    let source = r#"
(define (hello)
  (display "hello"))

(define (add a b)
  (+ a b))
"#;
    let symbols = parse_and_extract(source, Language::Scheme);
    assert!(
        symbols.len() >= 2,
        "Scheme should extract at least 2 symbols, got {}",
        symbols.len()
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function),
        "Scheme should extract function 'hello'"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
        "Scheme should extract function 'add'"
    );
}

#[test]
fn scheme_extracts_variable_define() {
    let source = "(define x 42)\n";
    let symbols = parse_and_extract(source, Language::Scheme);
    assert!(
        symbols.iter().any(|s| s.name.as_deref() == Some("x")),
        "Scheme should extract variable 'x'"
    );
}

#[test]
fn scheme_extracts_define_syntax() {
    let source = "(define-syntax my-macro\n  (syntax-rules () ()))\n";
    let symbols = parse_and_extract(source, Language::Scheme);
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("my-macro")),
        "Scheme should extract define-syntax 'my-macro'"
    );
}

#[test]
fn racket_extracts_definitions() {
    let source = r#"
#lang racket

(define (hello)
  (displayln "hello"))

(define (add a b)
  (+ a b))
"#;
    let symbols = parse_and_extract(source, Language::Racket);
    assert!(
        symbols.len() >= 2,
        "Racket should extract at least 2 symbols, got {}",
        symbols.len()
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("hello") && s.symbol_type == SymbolType::Function),
        "Racket should extract function 'hello'"
    );
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("add") && s.symbol_type == SymbolType::Function),
        "Racket should extract function 'add'"
    );
}

#[test]
fn racket_extracts_struct() {
    let source = "#lang racket\n(struct point (x y))\n";
    let symbols = parse_and_extract(source, Language::Racket);
    assert!(
        symbols
            .iter()
            .any(|s| s.name.as_deref() == Some("point") && s.symbol_type == SymbolType::Struct),
        "Racket should extract struct 'point'"
    );
}

#[test]
fn elm_extracts_functions_and_types() {
    let source = r#"
module Main exposing (main)

type alias Model =
    { count : Int
    }

type Msg
    = Increment
    | Decrement

main =
    text "hello"
"#;
    let symbols = parse_and_extract(source, Language::Elm);
    assert!(!symbols.is_empty(), "Elm should extract symbols");
}

#[test]
fn glsl_extracts_functions() {
    let source = r#"
struct Light {
    vec3 position;
    vec3 color;
};

void main() {
    gl_FragColor = vec4(1.0);
}
"#;
    let symbols = parse_and_extract(source, Language::Glsl);
    assert!(!symbols.is_empty(), "GLSL should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "GLSL should extract functions"
    );
}

#[test]
fn hlsl_extracts_functions() {
    let source = r#"
struct VS_OUTPUT {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

float4 main(float4 pos : SV_Position) : SV_Target {
    return float4(1, 0, 0, 1);
}
"#;
    let symbols = parse_and_extract(source, Language::Hlsl);
    assert!(!symbols.is_empty(), "HLSL should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "HLSL should extract functions"
    );
}

// ── Tier 2B symbol extraction tests ─────────────────

#[test]
fn svelte_extracts_blocks() {
    let source = r#"
<script>
  let count = 0;
  function increment() { count += 1; }
</script>

<button on:click={increment}>
  Count: {count}
</button>

<style>
  button { color: red; }
</style>
"#;
    let symbols = parse_and_extract(source, Language::Svelte);
    assert!(!symbols.is_empty(), "Svelte should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "Svelte should extract blocks"
    );
}

#[test]
fn astro_extracts_frontmatter_and_elements() {
    let source = r#"---
const title = "Hello";
const items = [1, 2, 3];
---
<html>
<head><title>{title}</title></head>
<body>
  <h1>{title}</h1>
</body>
</html>
"#;
    let symbols = parse_and_extract(source, Language::Astro);
    assert!(!symbols.is_empty(), "Astro should extract symbols");
}

#[test]
fn makefile_extracts_rules_and_variables() {
    let source = r#"
CC = gcc
CFLAGS = -Wall

all: build

build:
	$(CC) $(CFLAGS) -o main main.c

clean:
	rm -f main
"#;
    let symbols = parse_and_extract(source, Language::Makefile);
    assert!(!symbols.is_empty(), "Makefile should extract symbols");
    assert!(
        symbols
            .iter()
            .any(|s| s.symbol_type == SymbolType::Function),
        "Makefile should extract rules as functions"
    );
}

#[test]
fn ini_extracts_sections() {
    let source = r#"
[database]
host = localhost
port = 5432

[server]
bind = 0.0.0.0
"#;
    let symbols = parse_and_extract(source, Language::Ini);
    assert!(!symbols.is_empty(), "INI should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Block),
        "INI should extract sections as blocks"
    );
}

#[test]
fn nginx_extracts_blocks() {
    let source = r#"
server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://backend;
    }
}
"#;
    let symbols = parse_and_extract(source, Language::Nginx);
    assert!(!symbols.is_empty(), "Nginx should extract symbols");
}

#[test]
fn prisma_extracts_models_and_enums() {
    let source = r#"
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id    Int     @id @default(autoincrement())
  email String  @unique
  name  String?
  posts Post[]
}

model Post {
  id       Int    @id @default(autoincrement())
  title    String
  author   User   @relation(fields: [authorId], references: [id])
  authorId Int
}

enum Role {
  USER
  ADMIN
}
"#;
    let symbols = parse_and_extract(source, Language::Prisma);
    assert!(!symbols.is_empty(), "Prisma should extract symbols");
    assert!(
        symbols.iter().any(|s| s.symbol_type == SymbolType::Struct),
        "Prisma should extract models as structs"
    );
}

/// Three levels of inline `mod` around a single function. Each nesting level
/// costs the shared depth budget, so the innermost item is the one at risk.
const NESTED_INLINE_MOD_SOURCE: &str = r#"
mod outer {
    mod middle {
        mod inner {
            fn buried() -> i32 {
                1
            }
        }
    }
}
"#;

#[test]
fn rust_nested_inline_mods_reach_the_innermost_item() {
    let symbols = parse_and_extract(NESTED_INLINE_MOD_SOURCE, Language::Rust);
    let named: Vec<(&str, SymbolType)> = symbols
        .iter()
        .filter_map(|s| s.name.as_deref().map(|n| (n, s.symbol_type)))
        .collect();

    // Presence first: the enclosing modules must be there, or an extractor that
    // returned nothing would satisfy the assertion below by accident.
    assert!(
        named.contains(&("outer", SymbolType::Module)),
        "expected the outermost module, got {named:?}"
    );
    assert!(
        named.contains(&("middle", SymbolType::Module)),
        "expected the second-level module, got {named:?}"
    );
    assert!(
        named.contains(&("inner", SymbolType::Module)),
        "expected the third-level module, got {named:?}"
    );
    assert!(
        named.contains(&("buried", SymbolType::Function)),
        "expected the function inside three levels of inline mod, got {named:?}"
    );
}
