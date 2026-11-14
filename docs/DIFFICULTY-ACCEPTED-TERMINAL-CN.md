# Accepted trajectory terminal datasets

`dycon difficulty-dataset resolved DATASET --source-run RUN --index-id INDEX
--rollout-list ACCEPTED.json --eligibility accepted_terminal` freezes exact
action-close samples for explicitly selected, completed workspace evaluations,
including resolved=false and reward=0. The caller supplies the final accepted
canonical selection, for example from the receipt-bound capture publication.
An explicit rollout selection is required. Subject, report, canonical
AgentCompleted, native alignment and uncensored suffix checks remain enabled.

Labels describe the observed agent endpoint. Unresolved trajectories do not
receive success remainder labels. Defaults remain resolved_only;
positive_reward continues to require strictly positive official reward.

Configure `paths.artifact_roots.difficulty_dataset` as an absolute storage root
on a data disk. Both `online-index` and `difficulty-dataset` write under
`ROOT/difficulty/datasets/ID`; the selection loader uses the same resolver.
If an ID exists in both the configured root and legacy project root, resolution
rejects ambiguity. Global owned_path rules are unchanged. Model/evaluation
output storage is unchanged; this option only controls datasets and indices.

The source hidden tensors remain in their published NAS shards; dataset
construction does not make an extra copy of them. Put TMPDIR and logs on the
data disk too. Use the same storage-root configuration when consuming the new
dataset. Other training entrypoints with legacy direct paths may need their own
storage integration before use; this construction does not refit estimators.
