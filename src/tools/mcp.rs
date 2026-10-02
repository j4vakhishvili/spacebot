//! MCP tool adapters that proxy calls to external MCP servers.

use crate::authorization::{GateDecision, ToolGate, ToolHints};
use crate::mcp::McpConnection;
use crate::tools::truncate_output;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub struct McpToolAdapter {
    server_name: String,
    tool_name: String,
    description: String,
    input_schema: Value,
    hints: ToolHints,
    connection: Arc<McpConnection>,
    /// Department policy check for this worker's requesters. Checked against
    /// the exact server and tool pair, never the flattened model-facing name,
    /// which can collide across servers.
    gate: Option<ToolGate>,
}

impl McpToolAdapter {
    pub fn new(
        server_name: String,
        tool: rmcp::model::Tool,
        connection: Arc<McpConnection>,
    ) -> Self {
        let input_schema = tool.schema_as_json_value();
        let description = tool
            .description
            .map(|description| description.into_owned())
            .unwrap_or_default();

        let hints = ToolHints::from_annotations(tool.annotations.as_ref());

        Self {
            server_name,
            tool_name: tool.name.into_owned(),
            description,
            input_schema,
            hints,
            connection,
            gate: None,
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn hints(&self) -> ToolHints {
        self.hints
    }

    /// Check every call against the department policy before it reaches the
    /// server.
    pub fn with_gate(mut self, gate: ToolGate) -> Self {
        self.gate = Some(gate);
        self
    }

    fn namespaced_name(&self) -> String {
        format!(
            "{}_{}",
            sanitize_tool_identifier(&self.server_name),
            sanitize_tool_identifier(&self.tool_name)
        )
    }

    fn collect_result_text(result: &rmcp::model::CallToolResult) -> String {
        let mut blocks = result
            .content
            .iter()
            .map(|content| match &content.raw {
                rmcp::model::RawContent::Text(text) => text.text.clone(),
                rmcp::model::RawContent::Resource(resource) => match &resource.resource {
                    rmcp::model::ResourceContents::TextResourceContents { text, .. } => {
                        text.clone()
                    }
                    _ => serde_json::to_string(&content.raw)
                        .unwrap_or_else(|_| "[unsupported resource content]".to_string()),
                },
                other => serde_json::to_string(other)
                    .unwrap_or_else(|_| "[unsupported mcp content]".to_string()),
            })
            .collect::<Vec<_>>();

        if let Some(structured_content) = &result.structured_content {
            blocks.push(structured_content.to_string());
        }

        if blocks.is_empty() {
            String::new()
        } else {
            blocks.join("\n")
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("MCP tool call failed: {0}")]
pub struct McpToolError(String);

#[derive(Debug, Serialize)]
pub struct McpToolOutput {
    pub result: String,
}

impl Tool for McpToolAdapter {
    const NAME: &'static str = "mcp_tool";

    type Error = McpToolError;
    type Args = Value;
    type Output = McpToolOutput;

    fn name(&self) -> String {
        self.namespaced_name()
    }

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: self.namespaced_name(),
            description: self.description.clone(),
            parameters: self.input_schema.clone(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Some(gate) = &self.gate {
            match gate.check(&self.server_name, &self.tool_name, self.hints, Some(&args)) {
                GateDecision::Allow => {}
                GateDecision::Deny { reason } => {
                    tracing::info!(
                        server = %self.server_name,
                        tool = %self.tool_name,
                        requesters = ?gate.requesters().snapshot(),
                        "mcp call refused by department policy"
                    );
                    return Err(McpToolError(format!("Not permitted: {reason}")));
                }
                GateDecision::RequireApproval { class } => {
                    // Approvals are not wired yet, so a gated call fails
                    // closed rather than running unapproved.
                    tracing::info!(
                        server = %self.server_name,
                        tool = %self.tool_name,
                        class = %class,
                        requesters = ?gate.requesters().snapshot(),
                        "mcp call needs approval; refused until approvals are available"
                    );
                    return Err(McpToolError(format!(
                        "Not run: `{}` on `{}` is a {class} and needs approval from an \
                         approver, which this assistant can't request yet. Ask an approver \
                         to make this change directly.",
                        self.tool_name, self.server_name
                    )));
                }
            }
        }

        let result = self
            .connection
            .call_tool(&self.tool_name, args)
            .await
            .map_err(|error| McpToolError(error.to_string()))?;

        let output_text = Self::collect_result_text(&result);
        let output_text = truncate_output(&output_text, crate::tools::MAX_TOOL_OUTPUT_BYTES);

        if result.is_error.unwrap_or(false) {
            let message = if output_text.is_empty() {
                format!(
                    "MCP server '{}' reported an error while calling '{}'",
                    self.server_name, self.tool_name
                )
            } else {
                output_text
            };
            return Err(McpToolError(message));
        }

        if output_text.is_empty() {
            return Ok(McpToolOutput {
                result: "[tool returned no content]".to_string(),
            });
        }

        Ok(McpToolOutput {
            result: output_text,
        })
    }
}

fn sanitize_tool_identifier(raw: &str) -> String {
    let mut value = raw
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();

    while value.contains("__") {
        value = value.replace("__", "_");
    }

    value = value.trim_matches('_').to_string().if_empty("mcp_tool");

    if value
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_digit())
    {
        value.insert(0, '_');
    }

    value
}

trait EmptyStringExt {
    fn if_empty(self, fallback: &str) -> String;
}

impl EmptyStringExt for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}
