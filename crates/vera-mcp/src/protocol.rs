//! JSON-RPC / MCP protocol types.
//!
//! Implements the core protocol types for the Model Context Protocol (MCP)
//! over JSON-RPC 2.0 with stdio transport. Messages are newline-delimited
//! JSON objects on stdin/stdout.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol version we support.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC error codes.
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

// ── Incoming messages ───────────────────────────────────────────

/// A JSON-RPC message received from the client.
///
/// Can be a request (has `id` + `method`), notification (has `method`, no `id`),
/// or an invalid message.
#[derive(Debug, Deserialize)]
pub struct RpcMessage {
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
}

// ── Outgoing messages ───────────────────────────────────────────

/// A JSON-RPC success response.
#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    pub result: Value,
}

/// A JSON-RPC error response.
#[derive(Debug, Serialize)]
pub struct RpcError {
    pub jsonrpc: &'static str,
    pub id: Value,
    pub error: RpcErrorBody,
}

/// The `error` field of a JSON-RPC error response.
#[derive(Debug, Serialize)]
pub struct RpcErrorBody {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcResponse {
    pub fn new(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result,
        }
    }
}

impl RpcError {
    pub fn new(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            error: RpcErrorBody {
                code,
                message: message.into(),
                data: None,
            },
        }
    }
}

// ── MCP-specific types ──────────────────────────────────────────

/// Server info returned during initialization.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Capabilities advertised by the server.
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
}

/// Tool-related capability.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    pub list_changed: bool,
}

/// The result of the `initialize` method.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

/// A tool definition advertised to the client.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// The result of `tools/list`.
#[derive(Debug, Serialize)]
pub struct ToolsListResult {
    pub tools: Vec<ToolDefinition>,
}

/// A content item in a tool result.
#[derive(Debug, Serialize)]
pub struct TextContent {
    pub r#type: &'static str,
    pub text: String,
}

/// The result of `tools/call`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    pub content: Vec<TextContent>,
    pub is_error: bool,
}

impl ToolCallResult {
    /// Create a success result with text content.
    pub fn success(text: impl Into<String>) -> Self {
        Self {
            content: vec![TextContent {
                r#type: "text",
                text: text.into(),
            }],
            is_error: false,
        }
    }

    /// Create a success result carrying an optional advisory notice.
    ///
    /// The notice is a separate trailing content block, so `content[0]` stays
    /// byte-identical to what [`ToolCallResult::success`] would have produced
    /// and clients that parse it as JSON are unaffected.
    pub fn success_with_notice(text: impl Into<String>, notice: Option<String>) -> Self {
        let mut result = Self::success(text);
        if let Some(notice) = notice {
            result.content.push(TextContent {
                r#type: "text",
                text: notice,
            });
        }
        result
    }

    /// Create an error result with text content.
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![TextContent {
                r#type: "text",
                text: text.into(),
            }],
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_response_serializes_correctly() {
        let resp = RpcResponse::new(serde_json::json!(1), serde_json::json!({"ok": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"ok\":true"));
    }

    #[test]
    fn rpc_error_serializes_correctly() {
        let err = RpcError::new(serde_json::json!(2), METHOD_NOT_FOUND, "not found");
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("\"code\":-32601"));
        assert!(json.contains("\"message\":\"not found\""));
    }

    #[test]
    fn tool_call_result_success() {
        let result = ToolCallResult::success("hello world");
        assert!(!result.is_error);
        assert_eq!(result.content[0].text, "hello world");
        assert_eq!(result.content[0].r#type, "text");
    }

    #[test]
    fn tool_call_result_notice_is_a_trailing_block() {
        let result = ToolCallResult::success_with_notice("[]", Some("warning: stale".to_string()));
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 2);
        assert_eq!(result.content[0].text, "[]");
        assert_eq!(result.content[1].text, "warning: stale");
    }

    #[test]
    fn tool_call_result_without_notice_has_one_block() {
        let result = ToolCallResult::success_with_notice("[]", None);
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text, "[]");
    }

    #[test]
    fn tool_call_result_error() {
        let result = ToolCallResult::error("something went wrong");
        assert!(result.is_error);
        assert_eq!(result.content[0].text, "something went wrong");
    }

    #[test]
    fn initialize_result_serializes() {
        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
            },
            server_info: ServerInfo {
                name: "vera".to_string(),
                version: "0.1.0".to_string(),
            },
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(json["serverInfo"]["name"], "vera");
        assert_eq!(json["capabilities"]["tools"]["listChanged"], false);
    }

    #[test]
    fn tool_definition_serializes() {
        let tool = ToolDefinition {
            name: "search_code".to_string(),
            description: "Search indexed code".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "search_code");
        assert_eq!(json["inputSchema"]["type"], "object");
    }
}
