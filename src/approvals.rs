//! Human approval for gated MCP tool calls.
//!
//! When the department policy says a call needs approval, the worker's tool
//! call opens a request here and waits. The request is posted into the
//! conversation that owns the worker; an approver answers with a button or by
//! typing `approve <id>` / `deny <id>`. Answers are intercepted by the
//! inbound router before any model sees them, and only the request's listed
//! approvers can decide it. Every request and its outcome is recorded in
//! `tool_approvals`.
//!
//! Anything other than an explicit approval fails closed: a denial, the
//! timeout, the waiting worker being cancelled, or a restart.

use crate::authorization::Requester;
use crate::config::HumanDef;
use crate::{ChannelId, MessageContent, OutboundResponse, WorkerId};

use anyhow::Context as _;
use sha2::{Digest as _, Sha256};
use sqlx::SqlitePool;
use tokio::sync::oneshot;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a request waits for an answer before it is denied.
pub const DEFAULT_APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);

/// Prefix of the button action ids this module owns.
const ACTION_PREFIX: &str = "approval";

/// Longest argument summary shown on a card and stored in the audit row.
const ARGS_SUMMARY_MAX_CHARS: usize = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalVerdict {
    Approve,
    Deny,
}

/// How a request ended, as seen by the waiting tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    Approved { by: String },
    Denied { by: String },
    Expired,
}

/// What a gated call asks approval for.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub agent_id: String,
    pub worker_id: WorkerId,
    pub channel_id: Option<ChannelId>,
    pub server: String,
    pub tool: String,
    pub class: String,
    pub args: serde_json::Value,
    pub requesters: Vec<Requester>,
    pub approvers: Vec<String>,
    pub timeout: Duration,
}

/// A request that is open and waiting.
#[derive(Debug, Clone)]
pub struct OpenedApproval {
    pub approval_id: String,
    pub args_summary: String,
}

struct PendingEntry {
    approvers: Vec<String>,
    server: String,
    tool: String,
    sender: oneshot::Sender<ApprovalOutcome>,
}

/// The result of someone answering a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerResult {
    /// The answer decided the request.
    Decided {
        verdict: ApprovalVerdict,
        server: String,
        tool: String,
    },
    /// The person isn't one of this request's approvers.
    NotAllowed,
    /// No open request has this id: already decided, expired, or unknown.
    NotOpen,
}

/// Open approval requests for one agent, and their durable record.
pub struct ApprovalBroker {
    pool: SqlitePool,
    pending: Mutex<HashMap<String, PendingEntry>>,
}

impl std::fmt::Debug for ApprovalBroker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ApprovalBroker")
            .finish_non_exhaustive()
    }
}

impl ApprovalBroker {
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Mark every request still pending as expired. Run once at startup: the
    /// workers that were waiting on them did not survive the restart.
    pub async fn expire_interrupted(&self) -> anyhow::Result<u64> {
        let result = sqlx::query(
            "UPDATE tool_approvals SET status = 'expired', reason = 'restart', \
             decided_at = CURRENT_TIMESTAMP WHERE status = 'pending'",
        )
        .execute(&self.pool)
        .await
        .context("failed to expire interrupted approvals")?;
        Ok(result.rows_affected())
    }

    /// Record a request and start waiting for its answer.
    pub async fn open(
        &self,
        request: &ApprovalRequest,
        humans: &[HumanDef],
    ) -> anyhow::Result<(OpenedApproval, oneshot::Receiver<ApprovalOutcome>)> {
        let approval_id = new_approval_id();
        let args_summary = summarize_args(&request.args);
        let args_sha256 = hex::encode(Sha256::digest(request.args.to_string().as_bytes()));
        let requesters = request
            .requesters
            .iter()
            .map(|requester| requester_label(requester, humans))
            .collect::<Vec<_>>();
        let expires_at =
            chrono::Utc::now() + chrono::Duration::from_std(request.timeout).unwrap_or_default();

        sqlx::query(
            r#"
            INSERT INTO tool_approvals
                (approval_id, agent_id, worker_id, channel_id, server, tool, approval_class,
                 args_summary, args_sha256, requesters, approvers, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&approval_id)
        .bind(&request.agent_id)
        .bind(request.worker_id.to_string())
        .bind(request.channel_id.as_deref())
        .bind(&request.server)
        .bind(&request.tool)
        .bind(&request.class)
        .bind(&args_summary)
        .bind(&args_sha256)
        .bind(serde_json::to_string(&requesters)?)
        .bind(serde_json::to_string(&request.approvers)?)
        .bind(expires_at)
        .execute(&self.pool)
        .await
        .context("failed to record approval request")?;

        let (sender, receiver) = oneshot::channel();
        self.lock_pending().insert(
            approval_id.clone(),
            PendingEntry {
                approvers: request.approvers.clone(),
                server: request.server.clone(),
                tool: request.tool.clone(),
                sender,
            },
        );
        Ok((
            OpenedApproval {
                approval_id,
                args_summary,
            },
            receiver,
        ))
    }

    /// Apply someone's answer. Only a listed approver can decide; the first
    /// valid answer wins.
    pub async fn answer(
        &self,
        approval_id: &str,
        verdict: ApprovalVerdict,
        answerer: &Requester,
    ) -> AnswerResult {
        let (entry, decided_by) = {
            let mut pending = self.lock_pending();
            let Some(entry) = pending.get(approval_id) else {
                return AnswerResult::NotOpen;
            };
            let decided_by = match answerer {
                Requester::Human { id } if entry.approvers.contains(id) => id.clone(),
                _ => return AnswerResult::NotAllowed,
            };
            let Some(entry) = pending.remove(approval_id) else {
                return AnswerResult::NotOpen;
            };
            (entry, decided_by)
        };
        let (status, outcome) = match verdict {
            ApprovalVerdict::Approve => (
                "approved",
                ApprovalOutcome::Approved {
                    by: decided_by.clone(),
                },
            ),
            ApprovalVerdict::Deny => (
                "denied",
                ApprovalOutcome::Denied {
                    by: decided_by.clone(),
                },
            ),
        };
        if let Err(error) = self
            .record_outcome(approval_id, status, Some(&decided_by), None)
            .await
        {
            tracing::warn!(%error, approval_id, "failed to record approval decision");
        }
        // The waiting call may have gone away in the meantime; the decision is
        // recorded either way.
        entry.sender.send(outcome).ok();
        AnswerResult::Decided {
            verdict,
            server: entry.server,
            tool: entry.tool,
        }
    }

    /// Close a request nobody decided: the wait timed out or the worker went
    /// away. Returns `false` when an answer already closed it.
    pub async fn close_undecided(&self, approval_id: &str, status: &str, reason: &str) -> bool {
        let removed = self.lock_pending().remove(approval_id).is_some();
        if removed
            && let Err(error) = self
                .record_outcome(approval_id, status, None, Some(reason))
                .await
        {
            tracing::warn!(%error, approval_id, "failed to record approval closure");
        }
        removed
    }

    async fn record_outcome(
        &self,
        approval_id: &str,
        status: &str,
        decided_by: Option<&str>,
        reason: Option<&str>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE tool_approvals SET status = ?, decided_by = ?, reason = ?, \
             decided_at = CURRENT_TIMESTAMP WHERE approval_id = ? AND status = 'pending'",
        )
        .bind(status)
        .bind(decided_by)
        .bind(reason)
        .bind(approval_id)
        .execute(&self.pool)
        .await
        .context("failed to update approval")?;
        Ok(())
    }

    fn lock_pending(&self) -> std::sync::MutexGuard<'_, HashMap<String, PendingEntry>> {
        match self.pending.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// Closes a request whose waiting call is dropped (the worker was cancelled
/// or timed out) before the request was decided.
pub struct PendingApprovalGuard {
    broker: Arc<ApprovalBroker>,
    approval_id: Option<String>,
}

impl PendingApprovalGuard {
    pub fn new(broker: Arc<ApprovalBroker>, approval_id: String) -> Self {
        Self {
            broker,
            approval_id: Some(approval_id),
        }
    }

    /// The request was settled through the normal path.
    pub fn disarm(&mut self) {
        self.approval_id = None;
    }
}

impl Drop for PendingApprovalGuard {
    fn drop(&mut self) {
        let Some(approval_id) = self.approval_id.take() else {
            return;
        };
        let broker = self.broker.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn(async move {
                    broker
                        .close_undecided(&approval_id, "cancelled", "worker stopped waiting")
                        .await;
                });
            }
            Err(_) => {
                broker.lock_pending().remove(&approval_id);
            }
        }
    }
}

/// What a worker's MCP tools need to ask for approval: where to post the
/// request and who is asking.
#[derive(Clone)]
pub struct ApprovalContext {
    pub broker: Arc<ApprovalBroker>,
    pub humans: Arc<arc_swap::ArcSwap<Vec<HumanDef>>>,
    pub agent_id: crate::AgentId,
    pub worker_id: WorkerId,
    pub channel_id: Option<ChannelId>,
    /// The message the request replies to. `None` means nobody can be asked.
    pub origin: Option<crate::InboundMessage>,
    pub messaging: Option<Arc<crate::messaging::MessagingManager>>,
    pub api_state: Option<Arc<crate::api::ApiState>>,
}

impl std::fmt::Debug for ApprovalContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ApprovalContext")
            .field("agent_id", &self.agent_id)
            .field("worker_id", &self.worker_id)
            .field("channel_id", &self.channel_id)
            .finish_non_exhaustive()
    }
}

impl ApprovalContext {
    /// Ask for approval and wait for the outcome. `Err` carries a
    /// model-readable reason the request couldn't be made at all.
    #[allow(clippy::too_many_arguments)]
    pub async fn request(
        &self,
        server: &str,
        tool: &str,
        class: &str,
        approvers: Vec<String>,
        args: &serde_json::Value,
        requesters: Vec<Requester>,
        timeout: Duration,
    ) -> std::result::Result<ApprovalOutcome, String> {
        let Some(origin) = self.origin.as_ref() else {
            return Err(
                "this work isn't attached to a conversation, so there is nobody to ask".into(),
            );
        };
        if approvers.is_empty() {
            return Err("nobody is configured to approve it".into());
        }
        let humans = self.humans.load();
        let request = ApprovalRequest {
            agent_id: self.agent_id.to_string(),
            worker_id: self.worker_id,
            channel_id: self.channel_id.clone(),
            server: server.to_string(),
            tool: tool.to_string(),
            class: class.to_string(),
            args: args.clone(),
            requesters,
            approvers,
            timeout,
        };
        let (opened, receiver) = self.broker.open(&request, &humans).await.map_err(|error| {
            tracing::warn!(%error, "failed to open approval request");
            "the approval request couldn't be recorded".to_string()
        })?;
        let mut guard = PendingApprovalGuard::new(self.broker.clone(), opened.approval_id.clone());

        let card = approval_card(&opened, &request, &humans);
        if let Err(error) = self.post(origin, card).await {
            tracing::warn!(%error, approval_id = %opened.approval_id, "failed to post approval request");
            guard.disarm();
            self.broker
                .close_undecided(&opened.approval_id, "cancelled", "request not delivered")
                .await;
            return Err("the approval request couldn't be posted to the conversation".into());
        }
        tracing::info!(
            approval_id = %opened.approval_id,
            server,
            tool,
            class,
            "waiting for approval"
        );

        let outcome = match tokio::time::timeout(timeout, receiver).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => ApprovalOutcome::Expired,
            Err(_) => {
                if self
                    .broker
                    .close_undecided(&opened.approval_id, "expired", "timeout")
                    .await
                {
                    let note = OutboundResponse::Text(format!(
                        "Approval request `{}` expired with no answer, so `{tool}` was not run.",
                        opened.approval_id
                    ));
                    if let Err(error) = self.post(origin, note).await {
                        tracing::debug!(%error, "failed to post approval expiry note");
                    }
                }
                ApprovalOutcome::Expired
            }
        };
        guard.disarm();
        Ok(outcome)
    }

    async fn post(
        &self,
        origin: &crate::InboundMessage,
        response: OutboundResponse,
    ) -> anyhow::Result<()> {
        // Portal delivery rides the SSE bus; the portal adapter can't render
        // buttons, so its people answer with the typed commands.
        if origin.adapter_key() == "portal" {
            let Some(api_state) = self.api_state.as_ref() else {
                anyhow::bail!("can't post approval: no portal event bus");
            };
            let text = match response {
                OutboundResponse::RichMessage { text, .. } | OutboundResponse::Text(text) => text,
                _ => anyhow::bail!("can't post approval: unsupported portal response"),
            };
            api_state
                .event_tx
                .send(crate::api::ApiEvent::OutboundMessage {
                    agent_id: self.agent_id.to_string(),
                    channel_id: origin.conversation_id.clone(),
                    text,
                })
                .map_err(|_| anyhow::anyhow!("can't post approval: no portal listeners"))?;
            return Ok(());
        }
        let Some(messaging) = self.messaging.as_ref() else {
            anyhow::bail!("can't post approval: messaging is not running");
        };
        messaging.respond(origin, response).await?;
        Ok(())
    }
}

/// Read an approval answer from an inbound message: a button click
/// (`approval:<id>:approve`) or a typed `approve <id>` / `deny <id>`.
pub fn parse_answer(content: &MessageContent) -> Option<(String, ApprovalVerdict)> {
    match content {
        MessageContent::Interaction { action_id, .. } => {
            let mut parts = action_id.splitn(3, ':');
            if parts.next()? != ACTION_PREFIX {
                return None;
            }
            let approval_id = parts.next()?.to_string();
            let verdict = match parts.next()? {
                "approve" => ApprovalVerdict::Approve,
                "deny" => ApprovalVerdict::Deny,
                _ => return None,
            };
            is_approval_id(&approval_id).then_some((approval_id, verdict))
        }
        MessageContent::Text(text) => {
            let text = text.trim().trim_matches('`');
            let mut words = text.split_whitespace();
            let verdict = match words.next()?.to_ascii_lowercase().as_str() {
                "approve" => ApprovalVerdict::Approve,
                "deny" => ApprovalVerdict::Deny,
                _ => return None,
            };
            let approval_id = words.next()?.trim_matches('`').to_ascii_lowercase();
            if words.next().is_some() {
                return None;
            }
            is_approval_id(&approval_id).then_some((approval_id, verdict))
        }
        _ => None,
    }
}

fn new_approval_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..10].to_string()
}

fn is_approval_id(value: &str) -> bool {
    value.len() == 10 && value.chars().all(|character| character.is_ascii_hexdigit())
}

fn summarize_args(args: &serde_json::Value) -> String {
    let text = crate::secrets::scrub::scrub_leaks(&args.to_string());
    if text.chars().count() <= ARGS_SUMMARY_MAX_CHARS {
        return text;
    }
    let truncated = text
        .chars()
        .take(ARGS_SUMMARY_MAX_CHARS)
        .collect::<String>();
    format!("{truncated}…")
}

/// A person's name for cards and audit rows.
pub fn requester_label(requester: &Requester, humans: &[HumanDef]) -> String {
    match requester {
        Requester::Human { id } => human_label(id, humans),
        Requester::Unknown {
            platform,
            sender_id,
        } => {
            format!("unregistered {platform} user {sender_id}")
        }
        Requester::Unattended => "scheduled or background work".to_string(),
    }
}

pub fn human_label(id: &str, humans: &[HumanDef]) -> String {
    humans
        .iter()
        .find(|human| human.id == id)
        .and_then(|human| human.display_name.clone())
        .unwrap_or_else(|| id.to_string())
}

/// The card posted into the conversation for an open request: Slack blocks,
/// Discord buttons, and a plain-text version with typed commands for every
/// other surface.
pub fn approval_card(
    opened: &OpenedApproval,
    request: &ApprovalRequest,
    humans: &[HumanDef],
) -> OutboundResponse {
    let approval_id = &opened.approval_id;
    let requesters = request
        .requesters
        .iter()
        .map(|requester| requester_label(requester, humans))
        .collect::<Vec<_>>()
        .join(", ");
    let approvers = request
        .approvers
        .iter()
        .map(|id| human_label(id, humans))
        .collect::<Vec<_>>()
        .join(", ");
    let minutes = request.timeout.as_secs().div_ceil(60);
    let text = format!(
        "Approval needed ({class}): `{tool}` on `{server}`\n\
         Requested by: {requesters}\n\
         Arguments: {args}\n\
         Can approve: {approvers}\n\
         Reply `approve {approval_id}` or `deny {approval_id}` within {minutes} min; \
         no answer means deny.",
        class = request.class,
        tool = request.tool,
        server = request.server,
        args = opened.args_summary,
    );
    let action = |verdict: &str| format!("{ACTION_PREFIX}:{approval_id}:{verdict}");
    let blocks = vec![
        serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": format!(
                    "*Approval needed* ({class}): `{tool}` on `{server}`\n*Requested by:* {requesters}\n*Can approve:* {approvers}",
                    class = request.class,
                    tool = request.tool,
                    server = request.server,
                ),
            },
        }),
        serde_json::json!({
            "type": "section",
            "text": { "type": "mrkdwn", "text": format!("```{}```", opened.args_summary) },
        }),
        serde_json::json!({
            "type": "actions",
            "block_id": format!("{ACTION_PREFIX}:{approval_id}"),
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": { "type": "plain_text", "text": "Approve" },
                    "action_id": action("approve"),
                    "value": action("approve"),
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": { "type": "plain_text", "text": "Deny" },
                    "action_id": action("deny"),
                    "value": action("deny"),
                },
            ],
        }),
        serde_json::json!({
            "type": "context",
            "elements": [{
                "type": "mrkdwn",
                "text": format!("Request `{approval_id}` · expires in {minutes} min · no answer means deny"),
            }],
        }),
    ];
    OutboundResponse::RichMessage {
        text,
        blocks,
        cards: Vec::new(),
        interactive_elements: vec![crate::InteractiveElements::Buttons {
            buttons: vec![
                crate::Button {
                    label: "Approve".into(),
                    custom_id: Some(action("approve")),
                    style: crate::ButtonStyle::Success,
                    url: None,
                },
                crate::Button {
                    label: "Deny".into(),
                    custom_id: Some(action("deny")),
                    style: crate::ButtonStyle::Danger,
                    url: None,
                },
            ],
        }],
        poll: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn broker() -> ApprovalBroker {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        ApprovalBroker::new(pool)
    }

    fn request() -> ApprovalRequest {
        ApprovalRequest {
            agent_id: "main".into(),
            worker_id: uuid::Uuid::new_v4(),
            channel_id: Some(Arc::from("slack:T1:C1")),
            server: "agmcp-google-ads".into(),
            tool: "update_campaign".into(),
            class: "campaign_mutation".into(),
            args: serde_json::json!({"campaignId": "42", "status": "PAUSED"}),
            requesters: vec![Requester::Human {
                id: "advertiser".into(),
            }],
            approvers: vec!["lead".into(), "root".into()],
            timeout: DEFAULT_APPROVAL_TIMEOUT,
        }
    }

    fn person(id: &str) -> Requester {
        Requester::Human { id: id.into() }
    }

    async fn status(broker: &ApprovalBroker, approval_id: &str) -> (String, Option<String>) {
        let row: (String, Option<String>) =
            sqlx::query_as("SELECT status, decided_by FROM tool_approvals WHERE approval_id = ?")
                .bind(approval_id)
                .fetch_one(&broker.pool)
                .await
                .unwrap();
        row
    }

    #[tokio::test]
    async fn only_listed_approvers_decide_and_the_first_answer_wins() {
        let broker = broker().await;
        let (opened, receiver) = broker.open(&request(), &[]).await.unwrap();
        let id = opened.approval_id.as_str();

        // The requester isn't an approver, and an unregistered sender never is.
        assert_eq!(
            broker
                .answer(id, ApprovalVerdict::Approve, &person("advertiser"))
                .await,
            AnswerResult::NotAllowed
        );
        let stranger = Requester::Unknown {
            platform: "slack".into(),
            sender_id: "U0".into(),
        };
        assert_eq!(
            broker.answer(id, ApprovalVerdict::Approve, &stranger).await,
            AnswerResult::NotAllowed
        );

        assert!(matches!(
            broker
                .answer(id, ApprovalVerdict::Approve, &person("lead"))
                .await,
            AnswerResult::Decided {
                verdict: ApprovalVerdict::Approve,
                ..
            }
        ));
        assert_eq!(
            receiver.await.unwrap(),
            ApprovalOutcome::Approved { by: "lead".into() }
        );
        assert_eq!(
            broker
                .answer(id, ApprovalVerdict::Deny, &person("root"))
                .await,
            AnswerResult::NotOpen
        );
        assert_eq!(
            status(&broker, id).await,
            ("approved".to_string(), Some("lead".to_string()))
        );
    }

    #[tokio::test]
    async fn denials_timeouts_and_restarts_are_recorded() {
        let broker = broker().await;
        let (denied, receiver) = broker.open(&request(), &[]).await.unwrap();
        broker
            .answer(&denied.approval_id, ApprovalVerdict::Deny, &person("root"))
            .await;
        assert_eq!(
            receiver.await.unwrap(),
            ApprovalOutcome::Denied { by: "root".into() }
        );
        assert_eq!(
            status(&broker, &denied.approval_id).await.0,
            "denied".to_string()
        );

        let (timed_out, _receiver) = broker.open(&request(), &[]).await.unwrap();
        assert!(
            broker
                .close_undecided(&timed_out.approval_id, "expired", "timeout")
                .await
        );
        assert_eq!(
            broker
                .answer(
                    &timed_out.approval_id,
                    ApprovalVerdict::Approve,
                    &person("lead")
                )
                .await,
            AnswerResult::NotOpen
        );
        assert_eq!(status(&broker, &timed_out.approval_id).await.0, "expired");

        let (interrupted, _receiver) = broker.open(&request(), &[]).await.unwrap();
        assert_eq!(broker.expire_interrupted().await.unwrap(), 1);
        assert_eq!(status(&broker, &interrupted.approval_id).await.0, "expired");
    }

    #[tokio::test]
    async fn a_dropped_wait_cancels_the_request() {
        let broker = Arc::new(broker().await);
        let (opened, _receiver) = broker.open(&request(), &[]).await.unwrap();
        drop(PendingApprovalGuard::new(
            broker.clone(),
            opened.approval_id.clone(),
        ));
        for _ in 0..50 {
            if status(&broker, &opened.approval_id).await.0 == "cancelled" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(status(&broker, &opened.approval_id).await.0, "cancelled");
        assert_eq!(
            broker
                .answer(
                    &opened.approval_id,
                    ApprovalVerdict::Approve,
                    &person("lead")
                )
                .await,
            AnswerResult::NotOpen
        );
    }

    #[test]
    fn answers_parse_from_buttons_and_typed_commands() {
        let click = MessageContent::Interaction {
            action_id: "approval:0123456789:approve".into(),
            block_id: None,
            values: Vec::new(),
            label: None,
            message_ts: None,
        };
        assert_eq!(
            parse_answer(&click),
            Some(("0123456789".into(), ApprovalVerdict::Approve))
        );
        assert_eq!(
            parse_answer(&MessageContent::Text("  Deny `ABCDEF0123` ".into())),
            Some(("abcdef0123".into(), ApprovalVerdict::Deny))
        );
        for text in [
            "approve",
            "approve the campaign",
            "approve 0123456789 please",
            "please approve 0123456789",
            "approve 01234",
        ] {
            assert_eq!(
                parse_answer(&MessageContent::Text(text.into())),
                None,
                "{text}"
            );
        }
        let other_click = MessageContent::Interaction {
            action_id: "ask:abcd:1".into(),
            block_id: None,
            values: Vec::new(),
            label: None,
            message_ts: None,
        };
        assert_eq!(parse_answer(&other_click), None);
    }

    #[test]
    fn argument_summaries_are_scrubbed_and_bounded() {
        let summary = summarize_args(&serde_json::json!({
            "key": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOP",
            "notes": "x".repeat(2_000),
        }));
        assert!(!summary.contains("sk-ant-api03"), "{summary}");
        assert!(summary.chars().count() <= ARGS_SUMMARY_MAX_CHARS + 1);
    }
}
