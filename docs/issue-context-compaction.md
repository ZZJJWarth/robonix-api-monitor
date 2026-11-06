# [RFC] Opt-in monitor for validating context compaction before upstreaming

## Problem

Pilot resends the full planning history on every step. Execution wrappers, repeated
fallback records, superseded snapshots, and duplicated catalog/manual text accumulate,
so prompt tokens grow roughly quadratically with the number of steps. In a long
physical robot task this reached **400 messages and about 120k prompt tokens in a
single request**; the task consumed **3.93M tokens** end-to-end.

Long prompts also interact badly with execution: provider-side stream timeouts and
occasional empty completions become more likely as the prompt grows.

## Proposal

Do not change Robonix yet. First, validate compaction with an **external, opt-in
monitor** that sits between Pilot and the model provider.

- The monitor is a loopback OpenAI-compatible proxy. Pilot points its
  `VLM_BASE_URL` at it; no Robonix source or manifest change is required beyond
  that URL.
- It records every request/response locally and can optionally compact recognized
  Pilot requests before forwarding (`--compact-context`).
- Control and treatment arms are the same proxy with compaction off/on, so token
  accounting is identical and only one variable changes.
- If community A/B runs show no behavioral regression, the compaction logic can be
  moved into Robonix (Pilot history write-back or a system-layer component; the
  placement is a follow-up decision).

The monitor, deployment README, and A/B protocol are in this repository:
`README.md`.

## What compaction does

For a recognized Pilot request, compaction folds the execution suffix into a
deterministic **execution memory** block:

- Keeps the system prompt, user/task text, current capability catalog, capability
  manuals, the task record, the latest live snapshot, the current task board, and
  the empty-slot evidence.
- Keeps every call and every result as ordered evidence, including failures and
  repeated feedback for the same call.
- Drops duplicate transport wrappers and narration, superseded task-record
  revisions, superseded dispatch records, and stale snapshots that a newer
  authoritative snapshot replaces.
- Makes **no additional model call** and does not rewrite instructions or add
  task answers. The transformation is deterministic for the same input.

Observed shape in the 400-message case: 400 messages become 9 messages. This is a
structural consolidation, not a 98% information deletion: the retained messages
include the catalog (~26k chars), two capability manuals, a compact execution-memory
block (~93k chars), the task record, the live snapshot, and the task board.

## Preliminary evidence and limitations

These numbers come from one internal physical robot scenario and **one model**
(DeepSeek flash). Treat them as a hypothesis that needs community validation.

- End-to-end task: 3,927,758 total tokens uncompressed vs 1,062,624 compressed
  (**-73%**). Wall time also dropped, but behavior differences between runs make
  the wall-time comparison noisy.
- Replaying recorded decision points: prompt tokens fell by about **64%**.
- Output quality replay, 10 samples per condition: uncompressed prompts passed
  8/10 at one decision point and 10/10 at another; compressed prompts passed 10/10
  at both. The only blanks occurred with the 400-message uncompressed prompt.
  All non-blank outputs were strict JSON with a valid RTDL envelope and only
  capabilities present in the prompt.
- Task-level samples are too few (two compressed runs and one uncompressed in the
  clean comparison) and were affected by provider slowdowns and timeouts.

What this does **not** establish: behavior on other embodiments, other models,
other providers, or tasks with different evidence patterns; statistical
non-inferiority; prompt-cache behavior across providers.

## How to test

Use the A/B protocol in `README.md`:

- Same model, task set, reset procedure, and external success criterion in both arms.
- At least 3 distinct tasks, at least 10 paired runs per arm, alternating arms.
- Compare task success, invalid-output rate, request count, prompt/completion/total
  tokens, and wall time.
- Suggested non-inferiority gate: success-rate 95% CI lower bound no more than
  5 percentage points below control; invalid-output upper bound no more than
  5 percentage points above control; median prompt tokens at least 30% lower.

Please report results with the task, embodiment, model/provider, number of runs,
the confidence intervals, and any timeout/empty-response observations. Negative
results are useful: the point of the monitor is to find out where compaction is
unsafe before it reaches the system layer.

## Risks

- Prompt-cache behavior differs across providers. Reordering or shrinking history
  may reduce cached prefixes in ways the token totals do not show.
- Compaction can retain stale evidence if the history does not contain a newer
  authoritative snapshot. The current implementation keeps the newest snapshot
  and records revision boundaries, but domain semantics still matter.
- Evaluation bias: model self-reports are not success criteria, and task difficulty
  varies between runs.
- Very long prompts can produce empty responses or stream timeouts; an A/B must
  count these as failures, not drop them.
- Compaction is currently tuned for the Pilot RTDL history format. Non-Pilot
  requests are forwarded unchanged, and the monitor refuses to compact requests
  it cannot recognize.

## Not included in this proposal

- Any change to Robonix, Pilot prompts, capabilities, or contracts.
- A claim that compaction is universally safe.
- A production gateway or a hosted service. The monitor is a local test tool that
  binds to `127.0.0.1` and stores prompts locally; records must not be published.

## Ask

Run the A/B on at least one task matching your embodiment and model, and post the
summary table plus the acceptance-gate outcome. If several independent setups show
non-inferior behavior, we can propose moving the compaction logic upstream with
these results as evidence.
