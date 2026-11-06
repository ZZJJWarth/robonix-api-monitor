# rbnx-api-monitor

A loopback OpenAI-compatible proxy that records Pilot traffic and can optionally
compact repeated execution history before forwarding. It is the opt-in test
vehicle for validating context compaction **without changing Robonix**.

- Listens on `127.0.0.1` only.
- Never edits Robonix, Pilot, capabilities, or model prompts on disk.
- Does not change the response format; it forwards the upstream bytes to Pilot.
- Records every request/response under a local `records/` directory.

## Non-goals

- Not a production gateway, load balancer, or secret manager.
- Not a model router; provider-native bridges exist only so the same proxy can be
  used with GPT, Gemini, Claude, and DeepSeek planning requests.
- Does not decide whether compaction is safe. The community A/B runs do that.

## Architecture

```
Pilot ──HTTP──> 127.0.0.1:<port> ──HTTP──> provider Chat Completions
                     │
                     ├─ records/<request-id>/request.json      (received + forwarded body)
                     ├─ records/<request-id>/response.body     (raw upstream bytes)
                     ├─ records/<request-id>/summary.json      (status, usage, durations)
                     └─ records/index.jsonl                    (one line per request)
```

Compaction runs only when both conditions hold: `--compact-context` is set and
the request is recognized as a Robonix Pilot request (system prompt contains
`You are the Robonix Pilot` and `## RTDL output`). Non-Pilot requests are
recorded and forwarded unchanged.

## Requirements

- Python 3.9 or newer.
- `pip install -r requirements.txt` (`jsonschema` is the only third-party package).
- A Pilot process that can start against a custom Chat Completions base URL.
- A provider API key configured in Pilot (the proxy forwards `Authorization`
  unchanged; it never stores the key in cleartext).

## Important: base URL format

Pilot must use the monitor origin **without** `/v1`:

```text
VLM_BASE_URL=http://127.0.0.1:54090
```

The monitor accepts only `/chat/completions` and `/models`. A request to
`/v1/chat/completions` returns 404. Put any provider path prefix in
`--upstream` instead, e.g. `--upstream https://api.deepseek.com/v1`.

## Quick start: control vs treatment

Both arms go through the monitor so token accounting is identical. The **only**
difference is `--compact-context`. Use different ports and records directories.

Control (record only, no compaction):

```bash
cd rbnx-api-monitor
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

export UPSTREAM=https://api.deepseek.com/v1     # replace with your provider base URL
python3 monitor.py --profile generic --no-compact-context --upstream "$UPSTREAM" \
  --port 54090 --records records/control
```

Treatment (record + compaction) in a second terminal:

```bash
cd rbnx-api-monitor
. .venv/bin/activate
export UPSTREAM=https://api.deepseek.com/v1
python3 monitor.py --profile generic --compact-context --upstream "$UPSTREAM" \
  --port 54091 --records records/treatment
```

Verify each proxy before pointing Pilot at it:

```bash
curl -s http://127.0.0.1:54090/healthz | python3 -m json.tool
curl -s http://127.0.0.1:54091/healthz | python3 -m json.tool
```

Expected differences: control has `"context_compaction": false`; treatment has
`"context_compaction": true`. Both report `"profile": "generic"`.

Then, for each run:

1. Set `VLM_BASE_URL` in the environment or manifest that launches Pilot to
   `http://127.0.0.1:54090` (control) or `http://127.0.0.1:54091` (treatment).
2. Restart Pilot so it picks up the new base URL.
3. Run the task with the same model, seed/reset state, and success criterion.

`start-generic.sh` is a convenience wrapper for the treatment command; it reads
`RBNX_MONITOR_UPSTREAM`, `RBNX_MONITOR_PORT`, and `RBNX_MONITOR_RECORDS`.

## Profiles and flags

`--profile` selects domain behavior. The repository ships one built-in profile,
`generic`, which enables no domain behavior:

| Behavior | `generic` |
|---|---|
| Observation hints | disabled |
| Empty inventory slots | none filtered |
| Claude guidance | not injected |
| DeepSeek domain reminder | not appended |

Protocol features (`--deepseek-rtdl`, `--responses-schema`,
`--refine-rtdl-schema`, provider bridges) are independent of the profile and
work under `generic`.

`auto` always resolves to `generic`. To enable domain behavior, pass a JSON
profile path with `--profile /path/to/profile.json`. A profile may define:

| Field | Type | Effect |
|---|---|---|
| `name` | string | Name shown by `/healthz`. |
| `hints` | list | Subset of `empty_observation_hint`, `uninspected_object_hint`, `explore_followup_hint`, `embedded_error_hint`, `holding_delivery_hint`, `repeat_failure_hint`. |
| `hint_notes` | object | Note templates. Required keys are `empty_observation`, `uninspected_object`, `explore_followup`, `embedded_error` (`{error}`), `holding_delivery` (`{count}`), `repeat_failure` (`{count}`), `repeat_failure_with_error` (`{count}`, `{error}`). |
| `contracts` | object | `observation` (list), `inspect`, `explore`, `withdraw`, `deposit`, `inventory`. Contract IDs are matched by suffix. |
| `empty_slot_items` | list | Item IDs treated as empty slots during compaction. |
| `inventory_schema` | object | `transaction_count_field`, `slots_field`, `slot_field`, `item_field`, `count_field`. |
| `claude_guidance` | string | Extra system text for recognized Pilot requests. |
| `claude_guidance_version` | string | Recorded with injected guidance. |
| `rtdl_domain_reminder` | string | Domain rules appended after the DeepSeek protocol reminder. |

A hint flag is accepted only when the selected profile enables that hint;
otherwise the monitor exits with an error. Hints are never silently ignored.

Minimal profile example:

```json
{
  "name": "demo",
  "hints": ["empty_observation_hint"],
  "hint_notes": {"empty_observation": "No match; absence is not established."},
  "contracts": {"observation": ["demo/scene/find"]},
  "empty_slot_items": ["demo:air"],
  "inventory_schema": {"transaction_count_field": "held_after"},
  "claude_guidance": null,
  "rtdl_domain_reminder": null
}
```

Fields may be omitted; omitted lists and objects default to empty and omitted
strings default to null.

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--compact-context` / `--no-compact-context` | off | Compact recognized Pilot history. The treatment arm uses `--compact-context`; control uses the default or `--no-compact-context`. |
| `--profile {auto,generic,PATH}` | `auto` | Domain behavior. Protocol handling is unaffected. |
| `--provider {custom,gpt,gemini,deepseek,claude,auto}` | `custom` | Enables provider-native bridges for matching model names. |
| `--deepseek-rtdl` | off | DeepSeek RTDL request adaptation. Protocol-level, allowed in any profile. |
| `--responses-schema` | off | Requires `--deepseek-rtdl`. Protocol-level. |
| `--refine-rtdl-schema` | off | Optional schema binding to the current capability catalog. |
| `--upstream URL` | `https://api.deepseek.com` | Provider origin; path prefixes belong here. |
| `--port N` | `54090` | Loopback port. |
| `--records DIR` | `./records` | Record directory (created with mode 0700). |

`start-monitor.sh` is the maintainers' launcher. It defaults to `--profile
generic` and enables compaction; set `RBNX_MONITOR_PROFILE` to a JSON path to add
domain behavior.

## Records and privacy

- Directories and files are created under `umask 077`; the records root is 0700.
- `Authorization` headers and secret-named fields are redacted in JSON files.
- **Prompt text, task text, capability catalogs, and observations are not
  redacted.** Records are sensitive. Never commit, attach, or publish `records/`.
- Keep records only as long as the A/B analysis needs them, then delete the
  directory. `.gitignore` already excludes `records/`.

Per-request files:

| File | Content |
|---|---|
| `request.json` | received body (if modified), forwarded body, optimization metadata |
| `response.json` | upstream status and selected response headers |
| `response.body` | raw upstream bytes (SSE stream or JSON) |
| `summary.json` | status, usage, durations, finish reasons, error type |
| `../index.jsonl` | one summary per request, used for arm summaries |

## A/B protocol

The goal is to show that compaction does not degrade task behavior before it is
moved into Robonix. Treat an A/B as a screen, not proof.

1. Use the same model, task set, reset procedure, and success criterion in both
   arms. Change only `--compact-context`.
2. Run at least 10 paired executions per arm and at least 3 distinct tasks.
   Alternate arms (control, treatment, control, treatment, ...) so provider
   slowdowns and cache drift hit both arms.
3. Evaluate success externally (world state, task criterion, or a script), not
   from the model's self-report.
4. Collect per arm: task success rate, invalid-output rate, request count,
   prompt/completion/total tokens, and wall-clock duration.

Suggested acceptance gate:

- Success rate: treatment 95% confidence-interval lower bound no more than
  5 percentage points below control.
- Invalid output: treatment upper bound no more than 5 percentage points above
  control. Invalid means empty assistant content, malformed JSON when JSON is
  required, an unknown capability, or an unparseable RTDL envelope.
- Prompt tokens: treatment median at least 30% lower than control.
- No increase in provider errors or stream timeouts.

With 10 runs per arm these intervals are wide. Report the interval, not just the
point estimate. If the gate is inconclusive, add tasks and runs rather than
declaring success.

Summarize the two arms:

```bash
python3 tools/records_summary.py \
  --records records/control=control \
  --records records/treatment=treatment
```

The table reports requests, prompt/completion/total tokens, compaction hits,
blank responses, errors, and mean/median duration. Task success is external and
must be added manually per run.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401` from upstream | Pilot's key is missing or invalid; the proxy only forwards it | Check the key in Pilot's environment/manifest, not in the monitor |
| `404` with `/v1/...` | `VLM_BASE_URL` includes `/v1` | Use `http://127.0.0.1:<port>` and put `/v1` in `--upstream` |
| `502` | Upstream unreachable, TLS failure, or wrong origin | Verify `--upstream`; check the monitor stderr |
| Pilot aborts near 30 s with no tokens | Provider is slow or an intermediate proxy buffers the response; Pilot's idle timeout fires | Raise Pilot's `vlm_idle_timeout`, use a faster model, or remove buffering proxies |
| Empty assistant content | Provider returned only reasoning/finish metadata, or a blank completion | Inspect `summary.json` (`finish_reasons`, `error_type`); retry; the production Pilot retries blanks |
| Port already in use | Another monitor or service owns the port | Use `--port` and update `VLM_BASE_URL` |
| Every request is 404 | Path not `/chat/completions` or `/models` | Use an OpenAI-compatible Chat Completions client |
| `413` | Body over 32 MiB | Raise the limit only if you trust the client; prefer smaller prompts |
| Records missing | `--records` directory unwritable, or the request was rejected before recording | Check monitor stderr and directory permissions |
| Token columns are zero and `missing_usage` is high | Provider omits `usage` in streamed responses | Have Pilot request stream usage, use a non-streaming probe, or read totals from the provider dashboard |

## Rollback

1. Stop the monitor process.
2. Restore Pilot's original `VLM_BASE_URL` (or remove the override).
3. Restart Pilot.

Robonix files were never modified, so no code rollback is needed.

## AI agent checklist

Run these steps in order. Stop on the first failed gate and report the command
output. Do not print or commit API keys or `records/`.

1. Preconditions

   ```bash
   python3 --version          # gate: >= 3.9
   cd rbnx-api-monitor
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Self-test

   ```bash
   python3 -m unittest discover -p 'test_*.py'
   ```

   Gate: final line `OK`. Expected count is at least 164 tests.

3. Start the control arm

   ```bash
   export UPSTREAM=<provider-base-url>        # path prefix included, e.g. .../v1
   python3 monitor.py --profile generic --no-compact-context --upstream "$UPSTREAM" \
     --port 54090 --records records/control > /tmp/rbnx-monitor-control.log 2>&1 &
   echo $! > /tmp/rbnx-monitor-control.pid
   curl -s http://127.0.0.1:54090/healthz
   ```

   Gate: `"ok": true`, `"profile": "generic"`, `"context_compaction": false`.

4. Point Pilot at control and run one task

   Set `VLM_BASE_URL=http://127.0.0.1:54090` in the Pilot environment/manifest,
   restart Pilot, run the task, and confirm:

   ```bash
   test -f records/control/index.jsonl
   tail -n 1 records/control/index.jsonl
   ```

   Gate: a summary line exists and `error_type` is null.

5. Switch to the treatment arm

   ```bash
   kill "$(cat /tmp/rbnx-monitor-control.pid)"
   python3 monitor.py --profile generic --compact-context --upstream "$UPSTREAM" \
     --port 54091 --records records/treatment > /tmp/rbnx-monitor-treatment.log 2>&1 &
   echo $! > /tmp/rbnx-monitor-treatment.pid
   curl -s http://127.0.0.1:54091/healthz
   ```

   Gate: `"profile": "generic"`, `"compact-context"` feature reports
   `"context_compaction": true`.

6. Point Pilot at treatment and repeat the same task

   Set `VLM_BASE_URL=http://127.0.0.1:54091`, restart Pilot, run the same task,
   and confirm `records/treatment/index.jsonl` grows.

7. Summarize

   ```bash
   python3 tools/records_summary.py \
     --records records/control=control \
     --records records/treatment=treatment
   ```

   Gate: both arms recorded requests with token usage. Compare against the A/B
   acceptance gate above.

8. Restore

   Stop the monitor, restore Pilot's original `VLM_BASE_URL`, restart Pilot.
   Delete `records/` when the analysis is complete.
