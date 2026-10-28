---
name: esp32-release-check
description: Runs a production-readiness review of ESP32 firmware — sdkconfig hardening, partition table and app-size headroom, OTA and rollback, secure boot, flash and NVS encryption, debug interfaces, logging and versioning — and returns a GO / NO-GO report with evidence. Use when the user asks "is this ready to ship", "release checklist", "production build", "prepare firmware release", "review sdkconfig", "check partition table", "OTA rollback", "secure boot", "flash encryption", "disable JTAG", "app partition too small", "/release-check", or before tagging a firmware release for ESP32, ESP32-S3, ESP32-C3, ESP32-C6 or other Espressif chips.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Release Check

A release gate. Read-only: produce findings; any change goes through a contract.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` and attach the output as release evidence.

| Need | MCP tool | CLI |
|------|----------|-----|
| Everything at once | `project_doctor` | `esp32wb doctor . [--target T]` |
| sdkconfig production review | `sdkconfig_review` | `esp32wb sdkconfig sdkconfig` |
| Partition table + app fit | `partition_check` | `esp32wb partitions partitions.csv --sdkconfig sdkconfig --bin build/<app>.bin [--flash-size 4M]` |
| Pins for the real target | `pin_audit` | `esp32wb pins .` |
| Stack sizes | `stack_check` | `esp32wb stacks .` |
| Security feature support per chip | `search_esp_docs` | `esp32wb docs search "secure boot v2" --target T` |

Run against the **release** configuration (e.g. `sdkconfig.defaults` +
`sdkconfig.defaults.release` / the CI release profile), not a developer's local `sdkconfig`.

## Checklist

Mark each `PASS`, `FAIL`, `N/A` or `UNKNOWN` with evidence.

### 1. Build and configuration
- [ ] `CONFIG_IDF_TARGET` matches production hardware; ESP-IDF version pinned.
- [ ] Release config reproducible from committed `sdkconfig.defaults*`.
- [ ] Clean build, no new warnings; app version set (`PROJECT_VER` / `version.txt`).
- [ ] `esp32wb sdkconfig` shows no debug options left on (heap poisoning
      comprehensive, `PANIC_PRINT_HALT`, GDB stub, verbose log default, assertions policy as intended).
- [ ] Bootloader/app log level appropriate for production.

### 2. Partitions and size
- [ ] `esp32wb partitions` passes: no overlaps, aligned offsets, fits flash size.
- [ ] App binary fits every app slot with headroom (team rule, e.g. ≥ 10–20 % free) to
      allow future OTA images.
- [ ] NVS partition size adequate; `nvs_keys` partition present if NVS encryption is used.
- [ ] Partition layout change vs the fielded version? Partition tables cannot be changed
      by a normal OTA — treat as HIGH risk.

### 3. OTA and rollback
- [ ] `otadata` + at least two OTA app slots (`ota_0`, `ota_1`) if OTA is a requirement.
- [ ] HTTPS with server certificate verification (bundle or pinned CA); no skip-verify.
- [ ] `CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE` decision documented; if enabled, the new
      app calls `esp_ota_mark_app_valid_cancel_rollback()` only after self-test passes.
- [ ] Anti-rollback (`CONFIG_BOOTLOADER_APP_ANTI_ROLLBACK`) secure version planned — it
      burns eFuses (irreversible).
- [ ] Full cycle tested on hardware: download → reboot → validate → forced-failure rollback.
- Details: [OTA playbook](references/ota_update.md).

### 4. Security
- [ ] Secure boot and flash encryption decision recorded (enabled, or explicit risk
      acceptance). Availability and scheme (e.g. Secure Boot V2 RSA vs ECDSA) depend on
      chip **and silicon revision** — confirm in the ESP-IDF security guide for the target.
- [ ] Flash encryption in **Release** mode for production (Development mode allows re-flashing plaintext).
- [ ] NVS encryption enabled if secrets are stored in NVS.
- [ ] JTAG / USB-Serial-JTAG debug and ROM download mode policy decided (secure download
      mode or disabled) — eFuse based, irreversible.
- [ ] No credentials or private keys in source, logs, or `sdkconfig.defaults`.
- Details: [security checklist](references/security_checklist.md).

### 5. Robustness
- [ ] Brownout detector enabled at an appropriate level.
- [ ] Task WDT enabled; panic behaviour (`CONFIG_ESP_SYSTEM_PANIC`) set to reboot.
- [ ] Core dump to flash enabled if field crash analysis is required (needs a `coredump` partition).
- [ ] `esp_reset_reason()` logged/reported at boot.
- [ ] Free heap / min free heap headroom measured under worst-case load.
- [ ] `esp32wb stacks .` and `esp32wb pins .` clean.

### 6. Evidence and traceability
- [ ] Mission marked complete; test plan results attached; known gaps listed.
- [ ] Release notes: version, changes, migration notes (NVS schema, partition changes).
- [ ] Artifacts archived: `.bin`, bootloader, partition table, **ELF** (needed to decode field crashes), `sdkconfig`.

## Report

```markdown
# Release Check — <project> <version> (target: <chip>, IDF <ver>)
## Verdict: GO | GO WITH ACCEPTED RISKS | NO-GO
## Tool evidence (doctor / sdkconfig / partitions JSON summaries)
## Blockers (FAIL items)
## Accepted risks (with owner)
## Unknowns (could not verify — e.g. eFuse state of production units)
## Checklist
```

## Rules

- NO-GO on: app does not fit an OTA slot, TLS verification disabled, debug-only
  config in release, missing rollback strategy for a fleet OTA product.
- Never enable secure boot, flash encryption or burn eFuses as part of a review —
  recommend, and require explicit user approval and a contract.
- Report chip-specific security support as **UNKNOWN** unless confirmed from
  Espressif docs for the exact chip revision.
