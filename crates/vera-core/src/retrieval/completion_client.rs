//! OpenAI-compatible completion client for query expansion.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_ALTERNATIVES: usize = 2;
const MAX_ALLOWED_ALTERNATIVES: usize = 8;
const DEFAULT_MAX_TOKENS: u32 = 16_384;
const MIN_MAX_TOKENS: u32 = 128;
const MAX_MAX_TOKENS: u32 = 65_536;

/// Configuration for the OpenAI-compatible completion client.
#[derive(Debug, Clone)]
pub struct CompletionClientConfig {
    pub base_url: String,
    pub model_id: String,
    pub api_key: String,
    pub timeout: Duration,
    pub max_alternatives: usize,
    /// Token budget for completion output. Reasoning models may need large
    /// budgets since they emit `reasoning_content` before the final answer.
    pub max_tokens: u32,
}

impl CompletionClientConfig {
    /// Returns true when the required env vars are present.
    pub fn is_configured() -> bool {
        let has_base = std::env::var("VERA_COMPLETION_BASE_URL")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let has_model = std::env::var("VERA_COMPLETION_MODEL_ID")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        has_base && has_model
    }

    /// Build config from env vars.
    ///
    /// Required: `VERA_COMPLETION_BASE_URL`, `VERA_COMPLETION_MODEL_ID`.
    /// Optional: `VERA_COMPLETION_API_KEY` (default `none`),
    /// `VERA_COMPLETION_TIMEOUT_SECS`, `VERA_COMPLETION_MAX_ALTERNATIVES`,
    /// `VERA_COMPLETION_MAX_TOKENS`.
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("VERA_COMPLETION_BASE_URL")
            .context("VERA_COMPLETION_BASE_URL not set")?
            .trim()
            .to_string();
        let model_id = std::env::var("VERA_COMPLETION_MODEL_ID")
            .context("VERA_COMPLETION_MODEL_ID not set")?
            .trim()
            .to_string();
        let api_key = std::env::var("VERA_COMPLETION_API_KEY")
            .unwrap_or_else(|_| "none".to_string())
            .trim()
            .to_string();

        if base_url.is_empty() {
            return Err(anyhow!("VERA_COMPLETION_BASE_URL is empty"));
        }
        if model_id.is_empty() {
            return Err(anyhow!("VERA_COMPLETION_MODEL_ID is empty"));
        }

        let timeout = std::env::var("VERA_COMPLETION_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let max_alternatives = std::env::var("VERA_COMPLETION_MAX_ALTERNATIVES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, MAX_ALLOWED_ALTERNATIVES))
            .unwrap_or(DEFAULT_MAX_ALTERNATIVES);
        let max_tokens = std::env::var("VERA_COMPLETION_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(MIN_MAX_TOKENS, MAX_MAX_TOKENS))
            .unwrap_or(DEFAULT_MAX_TOKENS);

        Ok(Self {
            base_url,
            model_id,
            api_key,
            timeout: Duration::from_secs(timeout),
            max_alternatives,
            max_tokens,
        })
    }

    fn endpoint_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }
}

/// OpenAI-compatible chat completion client for query expansion.
pub struct CompletionClient {
    client: reqwest::blocking::Client,
    config: CompletionClientConfig,
}

impl CompletionClient {
    /// Create a client if env vars are configured, otherwise return `Ok(None)`.
    pub fn from_env_if_configured() -> Result<Option<Self>> {
        if !CompletionClientConfig::is_configured() {
            return Ok(None);
        }
        Self::from_env().map(Some)
    }

    /// Create a client from env vars.
    pub fn from_env() -> Result<Self> {
        crate::init_tls();
        let config = CompletionClientConfig::from_env()?;
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client for completion")?;
        Ok(Self { client, config })
    }

    /// Generate alternative queries with optional codebase context from a BM25
    /// pre-filter. When `context_hints` is non-empty the LLM sees real symbol
    /// names and file paths from the index, producing more targeted rewrites.
    pub fn expand_query_with_context(
        &self,
        query: &str,
        context_hints: &[String],
    ) -> Result<Vec<String>> {
        let context_block = if context_hints.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nRelevant symbols and files found in the codebase:\n{}\n\n\
                 Use these as hints to generate more targeted rewrites.",
                context_hints.join("\n")
            )
        };

        let prompt = format!(
            "Original query: {query}{context_block}\n\n\
             Decompose this into {} targeted sub-queries that each search for a \
             different code location or concept needed to answer the original query. \
             Each sub-query should target specific symbols, types, or code patterns \
             rather than rephrasing the same idea.\n\n\
             Return ONLY a JSON array of strings.",
            self.config.max_alternatives
        );

        let request = ChatCompletionRequest {
            model: &self.config.model_id,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "You decompose code-search queries into targeted sub-queries. \
                              Each sub-query should find a different piece of code needed \
                              to answer the original question. Use concrete symbol names, \
                              type names, or file patterns when possible. \
                              Output only the requested JSON."
                        .to_string(),
                },
                ChatMessage {
                    role: "user",
                    content: prompt,
                },
            ],
            temperature: 0.2,
            max_tokens: self.config.max_tokens,
            response_format: ChatResponseFormat {
                kind: "json_object",
            },
            stream: false,
        };

        let response = self
            .client
            .post(self.config.endpoint_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .context("failed to call completion API")?;

        let status = response.status();
        if !status.is_success() {
            let message = response.text().unwrap_or_default();
            return Err(anyhow!(
                "completion API error (status {}): {}",
                status.as_u16(),
                message
            ));
        }

        let payload: serde_json::Value = response
            .json()
            .context("failed to parse completion API response")?;

        let choice = payload
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|c| c.first())
            .ok_or_else(|| anyhow!("completion response did not include choices"))?;

        let message = choice
            .get("message")
            .ok_or_else(|| anyhow!("completion response did not include message"))?;

        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        if content.is_empty() {
            let finish_reason = choice
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let has_reasoning = message
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .is_some_and(|v| !v.trim().is_empty());
            if has_reasoning {
                return Err(anyhow!(
                    "completion response had empty content with only reasoning_content \
                     (finish_reason={finish_reason}); increase VERA_COMPLETION_MAX_TOKENS \
                     or use a non-reasoning model"
                ));
            }
            return Err(anyhow!(
                "completion response did not include message content"
            ));
        }

        let candidates = parse_query_candidates(content, self.config.max_alternatives)?;
        if candidates.is_empty() {
            return Err(anyhow!(
                "completion response did not contain any query candidates"
            ));
        }

        Ok(candidates)
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
    response_format: ChatResponseFormat,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

fn parse_query_candidates(raw: &str, limit: usize) -> Result<Vec<String>> {
    // Try parsing the whole string as JSON.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        if let Some(parsed) = json_value_to_queries(&value) {
            return Ok(normalize_query_list(parsed, limit));
        }
    }

    // Try extracting a JSON array from within the response (e.g. fenced code blocks).
    // `find` and `rfind` resolve independently: a reply whose last `]` precedes its
    // first `[` must fall through to the error below, not panic on the slice.
    if let Some(candidate) = raw
        .find('[')
        .zip(raw.rfind(']'))
        .filter(|(start, end)| end >= start)
        .and_then(|(start, end)| raw.get(start..=end))
    {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(candidate) {
            if let Some(parsed) = json_value_to_queries(&value) {
                return Ok(normalize_query_list(parsed, limit));
            }
        }
    }

    let preview = raw.lines().take(3).collect::<Vec<_>>().join(" ");
    Err(anyhow!(
        "completion response is not a JSON array of query strings: {}",
        preview
    ))
}

fn json_value_to_queries(value: &serde_json::Value) -> Option<Vec<String>> {
    fn extract_array(value: &serde_json::Value) -> Option<Vec<String>> {
        let array = value.as_array()?;
        let mut items = Vec::with_capacity(array.len());
        for item in array {
            items.push(item.as_str()?.to_string());
        }
        Some(items)
    }

    if let Some(parsed) = extract_array(value) {
        return Some(parsed);
    }

    // Try common wrapper keys.
    let object = value.as_object()?;
    for key in ["rewrites", "queries", "candidates", "alternatives"] {
        if let Some(parsed) = object.get(key).and_then(extract_array) {
            return Some(parsed);
        }
    }

    None
}

fn normalize_query_list(candidates: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();

    for candidate in candidates {
        let query = normalize_query(&candidate);
        if query.is_empty() {
            continue;
        }
        let key = query.to_ascii_lowercase();
        if seen.insert(key) {
            normalized.push(query);
        }
        if normalized.len() >= limit {
            break;
        }
    }

    normalized
}

fn normalize_query(query: &str) -> String {
    query
        .trim()
        .trim_matches('"')
        .trim_matches('`')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_array() {
        let raw = r#"["auth token refresh", "jwt expiry validation", "session middleware"]"#;
        let parsed = parse_query_candidates(raw, 4).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0], "auth token refresh");
    }

    #[test]
    fn parse_json_object_with_known_key() {
        let raw = r#"{"rewrites":["auth token refresh", "jwt expiry validation"]}"#;
        let parsed = parse_query_candidates(raw, 4).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn parse_fenced_json() {
        let raw = "```json\n[\"auth\", \"token refresh\"]\n```";
        let parsed = parse_query_candidates(raw, 4).unwrap();
        assert_eq!(parsed, vec!["auth", "token refresh"]);
    }

    #[test]
    fn non_json_returns_error() {
        let raw = "1. auth token refresh\n2) jwt expiry handling";
        assert!(parse_query_candidates(raw, 4).is_err());
    }

    /// A reply whose last `]` sits before its first `[` used to panic on
    /// `&raw[start..=end]`; it must return the normal parse error instead.
    #[test]
    fn closing_bracket_before_opening_bracket_returns_error() {
        let raw = "array done] here comes [\"q1\"";
        assert!(parse_query_candidates(raw, 4).is_err());
    }

    #[test]
    fn normalize_deduplicates_case_insensitive() {
        let result = normalize_query_list(
            vec![
                "Auth Refresh".to_string(),
                "auth refresh".to_string(),
                "AUTH REFRESH".to_string(),
            ],
            5,
        );
        assert_eq!(result, vec!["Auth Refresh"]);
    }
}
