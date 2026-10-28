# Mission Template

> Copy this file to `missions/YYYY-MM-DD-short-name.md` (or run `esp32wb mission new`) and fill in each section. Keep every `##` heading: `esp32wb mission validate` checks them.

## Goal

[What needs to be accomplished. Be specific and measurable.]

## Board / Target

- **Chip** (`CONFIG_IDF_TARGET`): [esp32 / esp32s2 / esp32s3 / esp32c2 / esp32c3 / esp32c5 / esp32c6 / esp32h2 / esp32p4]
- **Module / variant**: [e.g. ESP32-S3-WROOM-1-N16R8 — flash/PSRAM type changes which pins are reserved]
- **Board**: [DevKitC / custom / etc.]
- **ESP-IDF version**: [e.g., v5.2]

## Constraints

- [Hardware constraints — available pins, peripherals, power budget]
- [Software constraints — RTOS priorities, memory budget, latency requirements]
- [Project constraints — backward compatibility, API stability, timeline]

## Files in Scope

| File | Action | Notes |
|------|--------|-------|
| `path/to/file` | CREATE / MODIFY / DELETE | [Brief description] |

## Design Notes

[Architecture decisions, trade-offs, alternative approaches considered and rejected.]

## Acceptance Criteria

- [ ] [Specific, testable criterion]
- [ ] [Another criterion]
- [ ] All existing tests pass.
- [ ] New tests cover the change.
- [ ] Pin audit passes for the target chip (`esp32wb pins .`) if GPIO changes.

## Test Plan

> For non-trivial changes, write a full test plan (see the `esp32-test-plan` skill).

- [ ] [Test case 1]
- [ ] [Test case 2]
- [ ] No-hardware tests pass.
- [ ] Hardware tests pass (if applicable, mark with `@pytest.mark.hardware`).

## Known Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| [Description] | LOW/MED/HIGH | [How to handle] |

## Current Status

- **State**: NOT STARTED / IN PROGRESS / BLOCKED / REVIEW / DONE
- **Last updated**: [date]
- **Completed items**: [list]
- **Blocking issues**: [list]

## Next Step

[The single most important next action.]

## History

| Date | Action | Notes |
|------|--------|-------|
| [date] | Created | Initial mission |
