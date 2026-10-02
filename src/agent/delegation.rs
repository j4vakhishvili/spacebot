//! Cross-agent delegation: a channel runs a worker in another agent's context.
//!
//! The delegating channel owns the worker. It shows in that channel's status
//! block, takes follow-ups and cancellation there, and its completion
//! retriggers that channel, so events, the worker registry and the durable
//! worker record stay with the owner. What the worker can do comes from the
//! specialist: its MCP servers, skills, routing, workspace, sandbox, memory
//! and identity.

use crate::links::{AgentLink, LinkKind};
use crate::{AgentDeps, AgentId};

/// An agent this channel can hand work to.
#[derive(Clone)]
pub struct DelegationTarget {
    pub agent_id: AgentId,
    pub display_name: String,
    pub deps: AgentDeps,
}

impl std::fmt::Debug for DelegationTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DelegationTarget")
            .field("agent_id", &self.agent_id)
            .field("display_name", &self.display_name)
            .finish_non_exhaustive()
    }
}

/// Agents `owner` may delegate to: those it is the superior of through a
/// hierarchical link. Delegation only goes down the org chart, which keeps a
/// single coordinator and rules out loops between peers.
pub fn delegable_agent_ids(links: &[AgentLink], owner: &str) -> Vec<String> {
    let mut targets = Vec::new();
    for link in links {
        if link.kind == LinkKind::Hierarchical
            && link.from_agent_id == owner
            && link.to_agent_id != owner
            && !targets.contains(&link.to_agent_id)
        {
            targets.push(link.to_agent_id.clone());
        }
    }
    targets
}

/// Delegable agents with display names, for the spawn tool's description.
/// Agents that are linked but not running are left out.
pub async fn available_targets(owner: &AgentDeps) -> Vec<(String, String)> {
    let Some(api_state) = owner.api_state.as_ref() else {
        return Vec::new();
    };
    let ids = delegable_agent_ids(&owner.links.load(), &owner.agent_id);
    if ids.is_empty() {
        return Vec::new();
    }
    let registry = api_state.wake_registry.read().await;
    ids.into_iter()
        .filter(|id| registry.contains_key(id.as_str()))
        .map(|id| {
            let name = owner
                .agent_names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.clone());
            (id, name)
        })
        .collect()
}

/// Resolve a requested agent (id or display name) to a running agent this
/// channel may delegate to. The error text is written for the model.
pub async fn resolve_target(
    owner: &AgentDeps,
    requested: &str,
) -> std::result::Result<DelegationTarget, String> {
    let requested = requested.trim();
    let agent_id = owner
        .agent_names
        .iter()
        .find(|(id, name)| id.as_str() == requested || name.eq_ignore_ascii_case(requested))
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| requested.to_string());

    if agent_id == owner.agent_id.as_ref() {
        return Err(
            "That's this agent. Spawn the worker without `agent` to run it here.".to_string(),
        );
    }
    let delegable = delegable_agent_ids(&owner.links.load(), &owner.agent_id);
    if !delegable.contains(&agent_id) {
        return Err(format!(
            "Can't delegate to `{requested}`: it isn't one of this agent's specialists. \
             Delegable agents: {}.",
            if delegable.is_empty() {
                "none".to_string()
            } else {
                delegable.join(", ")
            }
        ));
    }
    let Some(api_state) = owner.api_state.as_ref() else {
        return Err("Delegation isn't available in this runtime.".to_string());
    };
    let Some(deps) = api_state
        .wake_registry
        .read()
        .await
        .get(agent_id.as_str())
        .cloned()
    else {
        return Err(format!(
            "Can't delegate to `{agent_id}`: that agent isn't running."
        ));
    };
    let display_name = owner
        .agent_names
        .get(&agent_id)
        .cloned()
        .unwrap_or_else(|| agent_id.clone());
    Ok(DelegationTarget {
        agent_id: deps.agent_id.clone(),
        display_name,
        deps,
    })
}

/// Dependencies for a worker owned by `owner` that works as `executor`.
///
/// Ownership plumbing (agent id, event buses, worker registry, database,
/// working memory, task store) stays with the owner. Capabilities come from
/// the executor. The authorization policy and humans are instance-wide, so
/// the owner's handles already are the executor's.
pub fn executor_deps(owner: &AgentDeps, executor: &AgentDeps) -> AgentDeps {
    let mut deps = owner.clone();
    deps.mcp_manager = executor.mcp_manager.clone();
    deps.runtime_config = executor.runtime_config.clone();
    deps.sandbox = executor.sandbox.clone();
    deps.memory_search = executor.memory_search.clone();
    deps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::links::LinkDirection;

    fn link(from: &str, to: &str, kind: LinkKind) -> AgentLink {
        AgentLink {
            from_agent_id: from.into(),
            to_agent_id: to.into(),
            direction: LinkDirection::TwoWay,
            kind,
        }
    }

    #[test]
    fn delegation_only_goes_down_hierarchical_links() {
        let links = vec![
            link("main", "ads", LinkKind::Hierarchical),
            link("main", "analytics", LinkKind::Hierarchical),
            link("main", "ads", LinkKind::Hierarchical),
            link("main", "peer", LinkKind::Peer),
            link("boss", "main", LinkKind::Hierarchical),
            link("ads", "analytics", LinkKind::Hierarchical),
        ];
        assert_eq!(
            delegable_agent_ids(&links, "main"),
            vec!["ads".to_string(), "analytics".to_string()]
        );
        // Specialists can delegate only to their own subordinates, and a
        // subordinate can never delegate back up.
        assert_eq!(
            delegable_agent_ids(&links, "ads"),
            vec!["analytics".to_string()]
        );
        assert!(delegable_agent_ids(&links, "analytics").is_empty());
    }
}
