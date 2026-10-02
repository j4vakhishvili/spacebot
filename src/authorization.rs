//! Department tool policy: who may make an agent call which MCP tool.
//!
//! Access follows the person, not the agent. Every worker carries the set of
//! people who directed it (its requesters), and each MCP call is checked
//! against that set: the effective access is the most restrictive outcome
//! across all of them, so a delegated or redirected worker never gains access
//! its requesters lack.

use crate::InboundMessage;
use crate::config::{
    AuthorizationConfig, DepartmentPolicy, HumanDef, ToolAccess, ToolRuleClass, ToolRuleDef,
};

use arc_swap::ArcSwap;
use serde_json::Value;

use std::sync::{Arc, RwLock};

/// Sender id the runtime uses for cron, autonomy and other internal messages.
const SYSTEM_SENDER_ID: &str = "system";

/// Leading words that mark a tool as a lookup. Checked only when the name
/// carries no write word.
const READ_VERBS: &[&str] = &[
    "list", "get", "lookup", "inspect", "check", "search", "describe", "query", "run", "batch",
    "count", "fetch", "read", "view", "find", "show", "preview", "estimate",
];

/// Words that mark a tool as changing, sending or spending. Any one of them
/// anywhere in the name makes the tool a write, even after a read verb.
const WRITE_WORDS: &[&str] = &[
    "create",
    "update",
    "delete",
    "add",
    "remove",
    "move",
    "merge",
    "assign",
    "activate",
    "pause",
    "resume",
    "enable",
    "disable",
    "toggle",
    "duplicate",
    "rename",
    "cancel",
    "manage",
    "mark",
    "set",
    "mutate",
    "archive",
    "adjust",
    "edit",
    "upload",
    "send",
    "publish",
    "submit",
    "write",
    "patch",
    "insert",
    "upsert",
    "import",
    "launch",
    "approve",
    "revoke",
    "grant",
    "apply",
    "restore",
    "reset",
    "clear",
    "reply",
    "forward",
    "enrich",
    "verify",
];

/// The person (or absence of one) a piece of work is done for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Requester {
    /// A configured human, resolved from an authenticated platform identity.
    Human { id: String },
    /// A sender that matches no configured human.
    Unknown { platform: String, sender_id: String },
    /// Work nobody asked for directly: cron, autonomy, resumed workers.
    Unattended,
}

/// The requesters of one worker. Shared between the worker's MCP tools and
/// the channel's `route` tool, which adds whoever sends the worker follow-up
/// input.
#[derive(Debug, Clone)]
pub struct RequesterSet(Arc<RwLock<Vec<Requester>>>);

impl RequesterSet {
    pub fn new(requesters: Vec<Requester>) -> Self {
        let mut unique = Vec::with_capacity(requesters.len());
        for requester in requesters {
            if !unique.contains(&requester) {
                unique.push(requester);
            }
        }
        if unique.is_empty() {
            unique.push(Requester::Unattended);
        }
        Self(Arc::new(RwLock::new(unique)))
    }

    pub fn unattended() -> Self {
        Self::new(vec![Requester::Unattended])
    }

    /// Add requesters. Access only ever narrows as people are added, so a
    /// lower-privileged sender can't widen a worker someone else started.
    pub fn extend(&self, requesters: &[Requester]) {
        let mut guard = match self.0.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for requester in requesters {
            if !guard.contains(requester) {
                guard.push(requester.clone());
            }
        }
    }

    pub fn snapshot(&self) -> Vec<Requester> {
        match self.0.read() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

/// Resolve the sender of an inbound message for authorization.
///
/// Only authenticated platform ids select a human. Email and id matches are
/// deliberately not used here: on the portal and webhook adapters the sender
/// is chosen by the client, so it can't identify anyone.
pub fn resolve_requester(
    message: &InboundMessage,
    humans: &[HumanDef],
    authorization: &AuthorizationConfig,
) -> Requester {
    if message.sender_id == SYSTEM_SENDER_ID || message.source == SYSTEM_SENDER_ID {
        return Requester::Unattended;
    }
    let platform = message.source.split(':').next().unwrap_or(&message.source);
    let human = match platform {
        "slack" => humans
            .iter()
            .find(|human| human.slack_id.as_deref() == Some(message.sender_id.as_str())),
        "discord" => humans
            .iter()
            .find(|human| human.discord_id.as_deref() == Some(message.sender_id.as_str())),
        "telegram" => humans
            .iter()
            .find(|human| human.telegram_id.as_deref() == Some(message.sender_id.as_str())),
        "portal" => authorization
            .portal_human
            .as_deref()
            .and_then(|portal_human| humans.iter().find(|human| human.id == portal_human)),
        _ => None,
    };
    match human {
        Some(human) => Requester::Human {
            id: human.id.clone(),
        },
        None => Requester::Unknown {
            platform: platform.to_string(),
            sender_id: message.sender_id.clone(),
        },
    }
}

/// Resolve every distinct sender in a batch of inbound messages. System
/// messages are skipped; `None` means the batch had no human input.
pub fn resolve_batch_requesters(
    messages: &[InboundMessage],
    humans: &[HumanDef],
    authorization: &AuthorizationConfig,
) -> Option<Vec<Requester>> {
    let mut requesters = Vec::new();
    for message in messages.iter().filter(|message| message.source != "system") {
        let requester = resolve_requester(message, humans, authorization);
        if !requesters.contains(&requester) {
            requesters.push(requester);
        }
    }
    (!requesters.is_empty()).then_some(requesters)
}

/// Hints an MCP server publishes about a tool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolHints {
    pub read_only: Option<bool>,
    pub destructive: Option<bool>,
}

impl ToolHints {
    pub fn from_annotations(annotations: Option<&rmcp::model::ToolAnnotations>) -> Self {
        annotations.map_or_else(Self::default, |annotations| Self {
            read_only: annotations.read_only_hint,
            destructive: annotations.destructive_hint,
        })
    }
}

/// What a tool call does, as far as the policy is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolClass {
    Read,
    /// A write; `class` names it when a tool rule did, so departments can
    /// require approval for that class specifically.
    Write {
        class: Option<String>,
    },
    /// Refused for everyone, admins included.
    Deny,
}

/// Classify one MCP call. In order: the first matching `[[tool_rules]]`
/// entry, the server's own annotations, then the verb heuristic. Anything
/// unrecognised is a write.
///
/// `args` is `None` when classifying a tool before any call (to decide
/// whether to offer it at all); dry-run arguments then can't apply.
pub fn classify(
    rules: &[ToolRuleDef],
    server: &str,
    tool: &str,
    hints: ToolHints,
    args: Option<&Value>,
) -> ToolClass {
    let tool_lower = tool.to_ascii_lowercase();
    let words = tool_words(&tool_lower);
    for rule in rules {
        if !rule_matches(rule, server, &tool_lower, &words) {
            continue;
        }
        if is_dry_run(rule, args) {
            return ToolClass::Read;
        }
        return match &rule.class {
            ToolRuleClass::Read => ToolClass::Read,
            ToolRuleClass::Write => ToolClass::Write { class: None },
            ToolRuleClass::Deny => ToolClass::Deny,
            ToolRuleClass::Named(class) => ToolClass::Write {
                class: Some(class.clone()),
            },
        };
    }
    if hints.read_only == Some(true) {
        return ToolClass::Read;
    }
    if hints.destructive == Some(true) {
        return ToolClass::Write { class: None };
    }
    if words.iter().any(|word| WRITE_WORDS.contains(word)) {
        return ToolClass::Write { class: None };
    }
    match words.first() {
        Some(first) if READ_VERBS.contains(first) => ToolClass::Read,
        _ => ToolClass::Write { class: None },
    }
}

fn tool_words(tool_lower: &str) -> Vec<&str> {
    tool_lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect()
}

fn rule_matches(rule: &ToolRuleDef, server: &str, tool_lower: &str, words: &[&str]) -> bool {
    if !matches_any(&rule.servers, server) {
        return false;
    }
    if !rule.tools.is_empty() && !matches_any(&rule.tools, tool_lower) {
        return false;
    }
    if !rule.verbs.is_empty()
        && !rule
            .verbs
            .iter()
            .any(|verb| words.contains(&verb.to_ascii_lowercase().as_str()))
    {
        return false;
    }
    if !rule.entities.is_empty()
        && !rule
            .entities
            .iter()
            .any(|entity| tool_lower.contains(&entity.to_ascii_lowercase()))
    {
        return false;
    }
    true
}

fn is_dry_run(rule: &ToolRuleDef, args: Option<&Value>) -> bool {
    let Some(Value::Object(args)) = args else {
        return false;
    };
    rule.unless_args
        .iter()
        .any(|(key, expected)| args.get(key) == Some(expected))
}

/// Glob match with `*` as the only wildcard, case-insensitive.
pub fn glob_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase();
    let value = value.to_ascii_lowercase();
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return value.is_empty();
    };
    if !pattern.contains('*') {
        return pattern == value;
    }
    let Some(mut rest) = value.strip_prefix(first) else {
        return false;
    };
    let remaining = parts.collect::<Vec<_>>();
    let Some((last, middle)) = remaining.split_last() else {
        return true;
    };
    for part in middle {
        match rest.find(part) {
            Some(index) => rest = &rest[index + part.len()..],
            None => return false,
        }
    }
    rest.len() >= last.len() && rest.ends_with(last)
}

fn matches_any(patterns: &[String], value: &str) -> bool {
    patterns.iter().any(|pattern| glob_matches(pattern, value))
}

/// The outcome of checking one call against the policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    Allow,
    /// The call is permitted only after an approver signs off.
    RequireApproval {
        class: String,
    },
    /// Refused. `reason` is written for the model to relay to the person.
    Deny {
        reason: String,
    },
}

impl GateDecision {
    fn severity(&self) -> u8 {
        match self {
            Self::Allow => 0,
            Self::RequireApproval { .. } => 1,
            Self::Deny { .. } => 2,
        }
    }
}

/// Decide a call for a set of requesters: the most restrictive individual
/// outcome wins.
pub fn decide(
    authorization: &AuthorizationConfig,
    humans: &[HumanDef],
    requesters: &[Requester],
    server: &str,
    tool: &str,
    class: &ToolClass,
) -> GateDecision {
    if !authorization.is_enabled() {
        return GateDecision::Allow;
    }
    if *class == ToolClass::Deny {
        return GateDecision::Deny {
            reason: format!(
                "`{tool}` on `{server}` is not available through the assistant for anyone. \
                 Do it directly in that service."
            ),
        };
    }
    let unattended = [Requester::Unattended];
    let requesters = if requesters.is_empty() {
        &unattended[..]
    } else {
        requesters
    };
    let several = requesters.len() > 1;
    requesters
        .iter()
        .map(|requester| {
            decide_for_requester(
                authorization,
                humans,
                requester,
                server,
                tool,
                class,
                several,
            )
        })
        .max_by_key(GateDecision::severity)
        .unwrap_or(GateDecision::Allow)
}

/// One grant that applies to a call: the access it gives and the write
/// classes it requires approval for.
struct Grant<'a> {
    access: ToolAccess,
    require_approval: &'a [String],
}

fn decide_for_requester(
    authorization: &AuthorizationConfig,
    humans: &[HumanDef],
    requester: &Requester,
    server: &str,
    tool: &str,
    class: &ToolClass,
    several: bool,
) -> GateDecision {
    let who = describe_requester(requester, humans, several);

    let (grants, admin_approval) = match requester {
        Requester::Unattended => {
            return match (class, authorization.unattended_access) {
                (ToolClass::Read, ToolAccess::Read | ToolAccess::Full) => GateDecision::Allow,
                (ToolClass::Read, ToolAccess::None) => GateDecision::Deny {
                    reason: format!(
                        "`{tool}` on `{server}` isn't available to {who}. \
                         Ask a person to run it."
                    ),
                },
                _ => GateDecision::Deny {
                    reason: format!(
                        "`{tool}` changes data on `{server}`, and {who} can only read. \
                         Ask a person with access to make this change."
                    ),
                },
            };
        }
        Requester::Human { id } => match humans.iter().find(|human| human.id == *id) {
            Some(human) if human.admin => {
                let approval_classes = if authorization.admin_requires_approval {
                    authorization
                        .departments
                        .iter()
                        .flat_map(|department| &department.policies)
                        .filter(|policy| policy_matches(policy, server, tool))
                        .flat_map(|policy| policy.require_approval.iter().cloned())
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                (Vec::new(), Some(approval_classes))
            }
            Some(human) => (
                grants_for_departments(authorization, &human.departments, server, tool),
                None,
            ),
            None => (default_department_grants(authorization, server, tool), None),
        },
        Requester::Unknown { .. } => (default_department_grants(authorization, server, tool), None),
    };

    if let Some(approval_classes) = admin_approval {
        return match class {
            ToolClass::Read => GateDecision::Allow,
            ToolClass::Write { class } => match approval_class(&approval_classes, class.as_deref())
            {
                Some(class) => GateDecision::RequireApproval { class },
                None => GateDecision::Allow,
            },
            ToolClass::Deny => GateDecision::Deny {
                reason: format!("`{tool}` on `{server}` is not available to anyone."),
            },
        };
    }

    let best_access = grants
        .iter()
        .map(|grant| grant.access)
        .max()
        .unwrap_or(ToolAccess::None);
    match class {
        ToolClass::Read if best_access >= ToolAccess::Read => GateDecision::Allow,
        ToolClass::Read => GateDecision::Deny {
            reason: format!(
                "`{tool}` on `{server}` isn't available to {who}. \
                 Ask someone with access to `{server}` to look this up."
            ),
        },
        ToolClass::Write { class } => {
            let full_grants = grants
                .iter()
                .filter(|grant| grant.access == ToolAccess::Full)
                .collect::<Vec<_>>();
            if full_grants.is_empty() {
                let detail = if best_access == ToolAccess::Read {
                    "can only read there"
                } else {
                    "has no access there"
                };
                return GateDecision::Deny {
                    reason: format!(
                        "`{tool}` changes data on `{server}`, and {who} {detail}. \
                         Ask someone with write access to `{server}` to make this change."
                    ),
                };
            }
            let mut required = None;
            for grant in full_grants {
                match approval_class(grant.require_approval, class.as_deref()) {
                    None => return GateDecision::Allow,
                    Some(class) => required = Some(class),
                }
            }
            match required {
                Some(class) => GateDecision::RequireApproval { class },
                None => GateDecision::Allow,
            }
        }
        ToolClass::Deny => GateDecision::Deny {
            reason: format!("`{tool}` on `{server}` is not available to anyone."),
        },
    }
}

/// The approval class a grant requires for a write, if any. `*` requires
/// approval for every write.
fn approval_class(require_approval: &[String], class: Option<&str>) -> Option<String> {
    if require_approval.iter().any(|entry| entry == "*") {
        return Some(class.unwrap_or("write").to_string());
    }
    let class = class?;
    require_approval
        .iter()
        .any(|entry| entry == class)
        .then(|| class.to_string())
}

fn policy_matches(policy: &DepartmentPolicy, server: &str, tool: &str) -> bool {
    matches_any(&policy.servers, server)
        && (policy.tools.is_empty() || matches_any(&policy.tools, tool))
}

fn grants_for_departments<'a>(
    authorization: &'a AuthorizationConfig,
    departments: &[String],
    server: &str,
    tool: &str,
) -> Vec<Grant<'a>> {
    // Within a department the first matching policy decides, so a narrow
    // entry listed before a broad one (`agmcp-gtm` read, then `agmcp-*` full)
    // narrows that server. Across departments the most permissive grant wins.
    let mut grants = departments
        .iter()
        .filter_map(|id| authorization.department(id))
        .filter_map(|department| {
            department
                .policies
                .iter()
                .find(|policy| policy_matches(policy, server, tool))
        })
        .map(|policy| Grant {
            access: policy.access,
            require_approval: &policy.require_approval,
        })
        .collect::<Vec<_>>();
    if departments.is_empty() {
        grants = default_department_grants(authorization, server, tool);
    }
    grants
}

fn default_department_grants<'a>(
    authorization: &'a AuthorizationConfig,
    server: &str,
    tool: &str,
) -> Vec<Grant<'a>> {
    match authorization.default_department.as_deref() {
        Some(id) => grants_for_departments(authorization, &[id.to_string()], server, tool),
        None => Vec::new(),
    }
}

fn describe_requester(requester: &Requester, humans: &[HumanDef], several: bool) -> String {
    let base = match requester {
        Requester::Unattended => "scheduled and background work".to_string(),
        Requester::Unknown { .. } => "people who aren't registered as staff".to_string(),
        Requester::Human { id } => match humans.iter().find(|human| human.id == *id) {
            Some(human) if !human.departments.is_empty() => {
                format!("the {} department", human.departments.join(" / "))
            }
            _ => "this person".to_string(),
        },
    };
    if several {
        format!("{base} (one of the people directing this task)")
    } else {
        base
    }
}

/// Per-worker policy check for MCP tools. Reads the live, hot-reloaded
/// policy on every call, so config changes apply to running workers.
#[derive(Debug, Clone)]
pub struct ToolGate {
    authorization: Arc<ArcSwap<AuthorizationConfig>>,
    humans: Arc<ArcSwap<Vec<HumanDef>>>,
    requesters: RequesterSet,
}

impl ToolGate {
    pub fn new(
        authorization: Arc<ArcSwap<AuthorizationConfig>>,
        humans: Arc<ArcSwap<Vec<HumanDef>>>,
        requesters: RequesterSet,
    ) -> Self {
        Self {
            authorization,
            humans,
            requesters,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.authorization.load().is_enabled()
    }

    pub fn requesters(&self) -> &RequesterSet {
        &self.requesters
    }

    /// Check one call. `args` is `None` when deciding whether to offer the
    /// tool at all.
    pub fn check(
        &self,
        server: &str,
        tool: &str,
        hints: ToolHints,
        args: Option<&Value>,
    ) -> GateDecision {
        let authorization = self.authorization.load();
        if !authorization.is_enabled() {
            return GateDecision::Allow;
        }
        let humans = self.humans.load();
        let class = classify(&authorization.tool_rules, server, tool, hints, args);
        decide(
            &authorization,
            &humans,
            &self.requesters.snapshot(),
            server,
            tool,
            &class,
        )
    }

    /// Whether a tool could ever be allowed for these requesters. Tools that
    /// can't are left out of the worker entirely, which also keeps them out
    /// of the model's context. A tool behind approval stays visible.
    pub fn is_offered(&self, server: &str, tool: &str, hints: ToolHints) -> bool {
        !matches!(
            self.check(server, tool, hints, None),
            GateDecision::Deny { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MessageContent;

    use serde_json::json;

    fn human(id: &str, departments: &[&str]) -> HumanDef {
        HumanDef {
            id: id.into(),
            display_name: None,
            role: None,
            bio: None,
            description: None,
            discord_id: None,
            telegram_id: None,
            slack_id: Some(format!("U{}", id.to_ascii_uppercase())),
            email: Some(format!("{id}@example.com")),
            departments: departments.iter().map(|value| value.to_string()).collect(),
            admin: false,
        }
    }

    fn policy(servers: &[&str], access: ToolAccess, require_approval: &[&str]) -> DepartmentPolicy {
        DepartmentPolicy {
            servers: servers.iter().map(|value| value.to_string()).collect(),
            tools: Vec::new(),
            access,
            require_approval: require_approval
                .iter()
                .map(|value| value.to_string())
                .collect(),
        }
    }

    fn rule(servers: &[&str], class: ToolRuleClass) -> ToolRuleDef {
        ToolRuleDef {
            servers: servers.iter().map(|value| value.to_string()).collect(),
            tools: Vec::new(),
            verbs: Vec::new(),
            entities: Vec::new(),
            unless_args: Vec::new(),
            class,
        }
    }

    /// The ad-guard rule: a mutation verb plus a campaign-level entity is a
    /// `campaign_mutation`, unless the call is a validate-only dry run.
    fn campaign_mutation_rule() -> ToolRuleDef {
        ToolRuleDef {
            verbs: [
                "create", "update", "delete", "mutate", "set", "pause", "resume", "enable",
                "disable", "archive", "remove", "adjust", "edit",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            entities: [
                "campaign",
                "budget",
                "insertion_order",
                "line_item",
                "order",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            unless_args: vec![
                ("validateOnly".into(), json!(true)),
                ("dryRun".into(), json!(true)),
            ],
            ..rule(
                &["agmcp-*"],
                ToolRuleClass::Named("campaign_mutation".into()),
            )
        }
    }

    fn instantly_deny_rule() -> ToolRuleDef {
        ToolRuleDef {
            tools: ["api_keys_*", "workspace_update", "dfy_orders_*"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            ..rule(&["instantly"], ToolRuleClass::Deny)
        }
    }

    fn ga4_report_jobs_rule() -> ToolRuleDef {
        ToolRuleDef {
            tools: vec!["create_report_task".into(), "create_audience_export".into()],
            ..rule(&["agmcp-ga4"], ToolRuleClass::Read)
        }
    }

    fn authorization() -> AuthorizationConfig {
        AuthorizationConfig {
            default_department: Some("everyone".into()),
            departments: vec![
                crate::config::DepartmentDef {
                    id: "everyone".into(),
                    approvers: Vec::new(),
                    policies: vec![policy(&["*"], ToolAccess::Read, &[])],
                },
                crate::config::DepartmentDef {
                    id: "sales".into(),
                    approvers: Vec::new(),
                    policies: vec![
                        policy(&["agmcp-*"], ToolAccess::Read, &[]),
                        policy(&["instantly"], ToolAccess::Full, &[]),
                    ],
                },
                crate::config::DepartmentDef {
                    id: "advertising".into(),
                    approvers: vec!["lead".into()],
                    policies: vec![
                        policy(
                            &["agmcp-gtm", "agmcp-ga4", "agmcp-search-console"],
                            ToolAccess::Read,
                            &[],
                        ),
                        policy(&["agmcp-*"], ToolAccess::Full, &["campaign_mutation"]),
                    ],
                },
                crate::config::DepartmentDef {
                    id: "measurement".into(),
                    approvers: Vec::new(),
                    policies: vec![policy(
                        &["agmcp-gtm", "agmcp-ga4", "agmcp-search-console"],
                        ToolAccess::Full,
                        &[],
                    )],
                },
                crate::config::DepartmentDef {
                    id: "back-office".into(),
                    approvers: Vec::new(),
                    policies: vec![policy(&["agmcp-*"], ToolAccess::None, &[])],
                },
            ],
            tool_rules: vec![
                instantly_deny_rule(),
                ga4_report_jobs_rule(),
                campaign_mutation_rule(),
            ],
            ..AuthorizationConfig::default()
        }
    }

    fn humans() -> Vec<HumanDef> {
        let mut admin = human("root", &[]);
        admin.admin = true;
        vec![
            human("seller", &["sales"]),
            human("advertiser", &["advertising"]),
            human("analyst", &["advertising", "measurement"]),
            human("clerk", &["back-office"]),
            human("lead", &["advertising"]),
            admin,
        ]
    }

    fn check(
        requesters: &[Requester],
        server: &str,
        tool: &str,
        args: Option<Value>,
    ) -> GateDecision {
        let authorization = authorization();
        let class = classify(
            &authorization.tool_rules,
            server,
            tool,
            ToolHints::default(),
            args.as_ref(),
        );
        decide(&authorization, &humans(), requesters, server, tool, &class)
    }

    fn person(id: &str) -> Requester {
        Requester::Human { id: id.into() }
    }

    fn stranger() -> Requester {
        Requester::Unknown {
            platform: "slack".into(),
            sender_id: "U0NOBODY".into(),
        }
    }

    fn message(source: &str, sender_id: &str) -> InboundMessage {
        InboundMessage {
            id: "message".into(),
            source: source.into(),
            adapter: None,
            conversation_id: "conversation".into(),
            sender_id: sender_id.into(),
            agent_id: None,
            content: MessageContent::Text("hello".into()),
            timestamp: chrono::Utc::now(),
            metadata: Default::default(),
            formatted_author: None,
        }
    }

    #[test]
    fn disabled_policy_allows_everything() {
        let authorization = AuthorizationConfig::default();
        let decision = decide(
            &authorization,
            &[],
            &[stranger()],
            "instantly",
            "api_keys_list",
            &ToolClass::Deny,
        );
        assert_eq!(decision, GateDecision::Allow);
    }

    #[test]
    fn reads_pass_for_everyone_with_read_access() {
        for requester in [
            person("seller"),
            person("advertiser"),
            stranger(),
            Requester::Unattended,
        ] {
            assert_eq!(
                check(
                    std::slice::from_ref(&requester),
                    "agmcp-google-ads",
                    "list_campaigns",
                    None
                ),
                GateDecision::Allow,
                "{requester:?}"
            );
        }
    }

    #[test]
    fn read_only_departments_never_write() {
        let decision = check(
            &[person("seller")],
            "agmcp-google-ads",
            "update_campaign",
            None,
        );
        assert!(
            matches!(decision, GateDecision::Deny { ref reason } if reason.contains("can only read")),
            "{decision:?}"
        );
        let decision = check(&[person("seller")], "agmcp-google-ads", "update_ad", None);
        assert!(
            matches!(decision, GateDecision::Deny { .. }),
            "{decision:?}"
        );
    }

    #[test]
    fn campaign_mutations_need_approval_for_full_access_departments() {
        assert_eq!(
            check(
                &[person("advertiser")],
                "agmcp-google-ads",
                "update_campaign",
                None
            ),
            GateDecision::RequireApproval {
                class: "campaign_mutation".into()
            }
        );
        assert_eq!(
            check(
                &[person("advertiser")],
                "agmcp-dv360",
                "set_line_item_budget",
                None
            ),
            GateDecision::RequireApproval {
                class: "campaign_mutation".into()
            }
        );
    }

    #[test]
    fn other_writes_on_full_access_servers_pass() {
        assert_eq!(
            check(
                &[person("advertiser")],
                "agmcp-google-ads",
                "update_ad",
                None
            ),
            GateDecision::Allow
        );
        assert_eq!(
            check(
                &[person("advertiser")],
                "agmcp-google-ads",
                "get_campaign",
                None
            ),
            GateDecision::Allow
        );
    }

    #[test]
    fn strict_boolean_dry_runs_are_reads() {
        assert_eq!(
            check(
                &[person("seller")],
                "agmcp-google-ads",
                "update_campaign",
                Some(json!({"validateOnly": true}))
            ),
            GateDecision::Allow
        );
        assert!(matches!(
            check(
                &[person("seller")],
                "agmcp-google-ads",
                "update_campaign",
                Some(json!({"validateOnly": "true"}))
            ),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn measurement_writes_follow_the_overlay_department() {
        // Advertising lists the measurement servers as read before its broad
        // full grant, so the narrower entry decides.
        let decision = check(&[person("advertiser")], "agmcp-gtm", "create_tag", None);
        assert!(
            matches!(decision, GateDecision::Deny { ref reason } if reason.contains("can only read")),
            "{decision:?}"
        );
        assert_eq!(
            check(&[person("analyst")], "agmcp-gtm", "create_tag", None),
            GateDecision::Allow
        );
        assert!(matches!(
            check(&[person("seller")], "agmcp-gtm", "create_tag", None),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn report_jobs_that_read_like_writes_are_reads() {
        assert_eq!(
            check(&[person("seller")], "agmcp-ga4", "create_report_task", None),
            GateDecision::Allow
        );
    }

    #[test]
    fn deny_tier_refuses_admins_too() {
        let decision = check(&[person("root")], "instantly", "api_keys_list", None);
        assert!(
            matches!(decision, GateDecision::Deny { ref reason } if reason.contains("for anyone")),
            "{decision:?}"
        );
        let decision = check(&[person("seller")], "instantly", "dfy_orders_create", None);
        assert!(matches!(decision, GateDecision::Deny { .. }));
    }

    #[test]
    fn admins_write_everywhere_but_keep_approval_classes() {
        assert_eq!(
            check(&[person("root")], "instantly", "activate_campaign", None),
            GateDecision::Allow
        );
        assert_eq!(
            check(&[person("root")], "agmcp-gtm", "create_tag", None),
            GateDecision::Allow
        );
        assert_eq!(
            check(
                &[person("root")],
                "agmcp-google-ads",
                "update_campaign",
                None
            ),
            GateDecision::RequireApproval {
                class: "campaign_mutation".into()
            }
        );
    }

    #[test]
    fn unknown_vocabulary_is_a_write() {
        assert!(matches!(
            check(
                &[person("seller")],
                "agmcp-google-ads",
                "campaigns_launch_now",
                None
            ),
            GateDecision::Deny { .. }
        ));
        assert!(matches!(
            check(&[person("seller")], "agmcp-google-ads", "frobnicate", None),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn unattended_work_reads_but_never_writes() {
        assert_eq!(
            check(&[Requester::Unattended], "agmcp-ga4", "run_report", None),
            GateDecision::Allow
        );
        let decision = check(
            &[Requester::Unattended],
            "instantly",
            "activate_campaign",
            None,
        );
        assert!(
            matches!(decision, GateDecision::Deny { ref reason } if reason.contains("scheduled and background work")),
            "{decision:?}"
        );
    }

    #[test]
    fn unknown_senders_get_the_default_department() {
        assert_eq!(
            check(&[stranger()], "instantly", "list_campaigns", None),
            GateDecision::Allow
        );
        assert!(matches!(
            check(&[stranger()], "instantly", "activate_campaign", None),
            GateDecision::Deny { .. }
        ));
    }

    #[test]
    fn departments_with_no_access_see_nothing() {
        let decision = check(
            &[person("clerk")],
            "agmcp-google-ads",
            "list_campaigns",
            None,
        );
        assert!(
            matches!(decision, GateDecision::Deny { .. }),
            "{decision:?}"
        );
    }

    #[test]
    fn several_requesters_get_the_intersection() {
        // A seller following up on an advertiser's worker can't widen it, and
        // the advertiser's access can't widen the seller's.
        let requesters = [person("advertiser"), person("seller")];
        let decision = check(&requesters, "agmcp-google-ads", "update_ad", None);
        assert!(
            matches!(decision, GateDecision::Deny { ref reason } if reason.contains("one of the people directing")),
            "{decision:?}"
        );
        let decision = check(&requesters, "instantly", "activate_campaign", None);
        assert!(
            matches!(decision, GateDecision::Deny { .. }),
            "{decision:?}"
        );
        assert_eq!(
            check(&requesters, "agmcp-google-ads", "list_campaigns", None),
            GateDecision::Allow
        );
    }

    #[test]
    fn requester_sets_only_grow() {
        let set = RequesterSet::new(vec![person("advertiser")]);
        set.extend(&[person("seller"), person("advertiser")]);
        assert_eq!(set.snapshot(), vec![person("advertiser"), person("seller")]);
        assert_eq!(
            RequesterSet::new(Vec::new()).snapshot(),
            vec![Requester::Unattended]
        );
    }

    #[test]
    fn annotations_classify_before_the_heuristic() {
        let read_only = ToolHints {
            read_only: Some(true),
            destructive: None,
        };
        assert_eq!(
            classify(&[], "server", "sync_everything", read_only, None),
            ToolClass::Read
        );
        let destructive = ToolHints {
            read_only: None,
            destructive: Some(true),
        };
        assert_eq!(
            classify(&[], "server", "list_and_purge", destructive, None),
            ToolClass::Write { class: None }
        );
        // An explicit rule still wins over the server's own hints.
        let rules = vec![ToolRuleDef {
            tools: vec!["sync_everything".into()],
            ..rule(&["server"], ToolRuleClass::Deny)
        }];
        assert_eq!(
            classify(&rules, "server", "sync_everything", read_only, None),
            ToolClass::Deny
        );
    }

    #[test]
    fn heuristic_reads_leading_verbs_and_flags_write_words() {
        let hints = ToolHints::default();
        assert_eq!(
            classify(&[], "s", "list_campaigns", hints, None),
            ToolClass::Read
        );
        assert_eq!(
            classify(&[], "s", "run_report", hints, None),
            ToolClass::Read
        );
        assert_eq!(
            classify(&[], "s", "batch_run_reports", hints, None),
            ToolClass::Read
        );
        assert_eq!(
            classify(&[], "s", "searchAnalytics", hints, None),
            ToolClass::Write { class: None }
        );
        assert_eq!(
            classify(&[], "s", "get_and_update_settings", hints, None),
            ToolClass::Write { class: None }
        );
        assert_eq!(
            classify(&[], "s", "campaigns-list", hints, None),
            ToolClass::Write { class: None }
        );
    }

    #[test]
    fn glob_matching() {
        assert!(glob_matches("agmcp-*", "agmcp-google-ads"));
        assert!(glob_matches("AGMCP-*", "agmcp-ga4"));
        assert!(!glob_matches("agmcp-*", "instantly"));
        assert!(glob_matches("*", "anything"));
        assert!(glob_matches("api_keys_*", "api_keys_list"));
        assert!(glob_matches("*_account_*", "create_account_now"));
        assert!(!glob_matches("*_account_*", "account"));
        assert!(glob_matches("exact", "EXACT"));
        assert!(!glob_matches("exact", "exactly"));
        assert!(glob_matches("a*b*c", "a-b-c"));
        assert!(!glob_matches("a*b*c", "a-c"));
    }

    #[test]
    fn requesters_resolve_from_authenticated_platform_ids_only() {
        let humans = humans();
        let mut authorization = authorization();
        assert_eq!(
            resolve_requester(&message("slack", "USELLER"), &humans, &authorization),
            person("seller")
        );
        // An email or a bare human id in the sender field is not an identity.
        assert!(matches!(
            resolve_requester(
                &message("webhook", "seller@example.com"),
                &humans,
                &authorization
            ),
            Requester::Unknown { .. }
        ));
        assert!(matches!(
            resolve_requester(&message("portal", "root"), &humans, &authorization),
            Requester::Unknown { .. }
        ));
        authorization.portal_human = Some("advertiser".into());
        assert_eq!(
            resolve_requester(&message("portal", "root"), &humans, &authorization),
            person("advertiser")
        );
        // Cron delivers through a platform adapter but sends as `system`.
        assert_eq!(
            resolve_requester(&message("slack", "system"), &humans, &authorization),
            Requester::Unattended
        );
    }

    #[test]
    fn batch_requesters_skip_system_messages() {
        let humans = humans();
        let authorization = authorization();
        let messages = vec![
            message("slack", "USELLER"),
            message("system", "system"),
            message("slack", "UADVERTISER"),
            message("slack", "USELLER"),
        ];
        assert_eq!(
            resolve_batch_requesters(&messages, &humans, &authorization),
            Some(vec![person("seller"), person("advertiser")])
        );
        assert_eq!(
            resolve_batch_requesters(&[message("system", "system")], &humans, &authorization),
            None
        );
    }

    #[test]
    fn gate_reads_the_live_policy() {
        let authorization = Arc::new(ArcSwap::from_pointee(authorization()));
        let humans = Arc::new(ArcSwap::from_pointee(humans()));
        let gate = ToolGate::new(
            authorization.clone(),
            humans,
            RequesterSet::new(vec![person("seller")]),
        );
        assert!(!gate.is_offered("agmcp-google-ads", "update_ad", ToolHints::default()));
        assert!(gate.is_offered("agmcp-google-ads", "list_ads", ToolHints::default()));
        authorization.store(Arc::new(AuthorizationConfig::default()));
        assert!(gate.is_offered("agmcp-google-ads", "update_ad", ToolHints::default()));
    }
}
