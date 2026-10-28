---
name: esp32-pr-prepare
description: Prepares a reviewable pull request for an ESP32 firmware change with summary, affected files, target chip, test evidence, static-check output, risk assessment, rollback plan and links to the mission and implementation contract. Use when the user says "prepare the PR", "write the PR description", "open a pull request", "ready for review", "/prcraft", "summarize my changes for review", or "mark the mission done" after implementing and testing an ESP-IDF change.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 PR Prepare

A PR is the audit record: what changed, why, how it was proven, what could go wrong.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output.

| Evidence | MCP tool | CLI |
|----------|----------|-----|
| Static checks on the final tree | `project_doctor` | `esp32wb doctor . [--target T]` |
| Pin audit (if GPIO changed) | `pin_audit` | `esp32wb pins .` |
| Partition / size (if app size or partitions changed) | `partition_check` | `esp32wb partitions partitions.csv --bin build/<app>.bin` |
| Mission completeness | `validate_mission` | `esp32wb mission validate missions/<file>.md` |

## Steps

### 1. Collect artifacts

From the mission and contract: summary, motivation, non-goals, affected files,
target chip(s), risks, rollback. If no contract exists, stop and say so — the
workflow requires one (a brief one is fine for tiny changes).

### 2. Check scope

Compare the actual diff (`git diff --stat <base>...HEAD`) with the contract's file list.
Unexpected files → explain them, update the contract, or split the PR.

### 3. Compile evidence

- Exact test commands and their output (pass/fail counts).
- `esp32wb doctor` summary; pin audit if GPIO changed.
- Hardware results: chip + board + IDF version, serial log excerpts, measurements.
- Skipped tests with reasons (e.g. hardware-only).
- **What was not tested** and why.

Never claim a test passed without its output.

### 4. Write the description

~~~markdown
## Summary
<what and why, one paragraph> — Mission: <link> · Contract: <link>

## Target
- Chip(s): <CONFIG_IDF_TARGET> · Board: <board> · ESP-IDF: <version>

## Changes
| File | Change | Description |
|------|--------|-------------|

### Architecture impact
- <tasks, memory, peripherals, new dependencies, Kconfig/NVS/partition changes>

## Testing
### Commands and results
```
<command>
<trimmed output>
```
- Passed: N · Failed: N · Skipped: N (reason)
### Static checks
<esp32wb doctor summary>
### Hardware validation
<board, steps, log excerpt> or "not performed — <reason>"
### Not tested
- <gap — reason>

## Risk assessment
| Risk | Level | Mitigation |
|------|-------|-----------|

## Rollback
<steps; NVS/OTA/partition implications>

## Checklist
- [ ] Contract accepted; scope matches diff
- [ ] Acceptance criteria met
- [ ] Tests added/updated; no regressions
- [ ] Pin audit clean for the target (if GPIO changed)
- [ ] Architecture review done (if structural change)
- [ ] Docs / pin map updated
- [ ] Mission updated to DONE with evidence
~~~

Use plain text markers (Passed/Failed/Skipped) rather than emoji so the description
renders consistently everywhere.

### 5. Readiness check

- [ ] CI would pass (same commands run locally).
- [ ] No debug code, stray logs, or test-only sdkconfig changes.
- [ ] No `TODO`/`FIXME` without an issue reference.
- [ ] Commit messages descriptive.
- [ ] Mission **Current Status** = REVIEW/DONE, **History** updated.

### 6. Hand-off

Output the PR description. Only run `git`/`gh` commands to create the PR if the user
asks; follow the repository's branch and attribution conventions.

## Rules

1. No PR without test evidence (manual tests count if documented).
2. Always include the risk assessment and rollback.
3. Always link the mission and contract.
4. Keep PRs focused; split when scope exceeded the contract.
