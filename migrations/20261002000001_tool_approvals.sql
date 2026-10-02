-- Human approvals for gated MCP tool calls, and the audit trail of every
-- decision.
--
-- A row is written when a worker's tool call needs approval and updated
-- once when it is approved, denied, expires, or the waiting worker goes away.
-- Rows still pending at startup belong to workers that no longer exist and
-- are marked expired.

CREATE TABLE IF NOT EXISTS tool_approvals (
    approval_id   TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    worker_id     TEXT NOT NULL,
    channel_id    TEXT,
    server        TEXT NOT NULL,
    tool          TEXT NOT NULL,
    approval_class TEXT NOT NULL,
    args_summary  TEXT NOT NULL,
    args_sha256   TEXT NOT NULL,
    requesters    TEXT NOT NULL,   -- JSON array of requester labels
    approvers     TEXT NOT NULL,   -- JSON array of human ids allowed to decide
    status        TEXT NOT NULL DEFAULT 'pending', -- pending | approved | denied | expired | cancelled
    decided_by    TEXT,            -- human id of the approver, for approved/denied
    reason        TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMP NOT NULL,
    decided_at    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_approvals_status
    ON tool_approvals(status, created_at);

CREATE INDEX IF NOT EXISTS idx_tool_approvals_worker
    ON tool_approvals(worker_id);
