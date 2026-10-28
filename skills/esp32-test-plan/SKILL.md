---
name: esp32-test-plan
description: Generates a layered test plan for an ESP32 firmware change — static checks, host-side pytest/C logic, ESP-IDF Unity tests on target, pytest-embedded integration — with positive, negative, boundary, concurrency and error cases and explicit pass/fail evidence. Use when the user asks for a "test plan", "how do I test this", "what tests do I need", "unit tests for this component", "pytest-embedded", "Unity test", "hardware-in-the-loop", "/test-plan", or after an implementation contract is accepted for ESP32, ESP32-S3, ESP32-C3, ESP32-C6 or other ESP-IDF projects.
license: MIT
metadata:
  source: esp32-workbench
  version: "0.2.0"
---

# ESP32 Test Plan

Every implementation contract needs a test plan; maximise tests that run without
hardware, and make hardware tests explicit.

## Tools

If the `esp32-workbench` MCP server is connected, call its tools; otherwise run the
CLI. If `esp32wb` is not on PATH, use
`uvx --from git+https://github.com/agodianel/esp32-claude-workbench esp32wb <args>`.
Add `--json` for machine-readable output (useful as CI evidence).

| Static layer (no hardware) | MCP tool | CLI |
|----------------------------|----------|-----|
| All static checks | `project_doctor` | `esp32wb doctor .` (non-zero exit on errors) |
| Pin audit for the target | `pin_audit` | `esp32wb pins .` |
| Stack sizes | `stack_check` | `esp32wb stacks .` |
| sdkconfig review | `sdkconfig_review` | `esp32wb sdkconfig sdkconfig` |
| Partition fit vs built app | `partition_check` | `esp32wb partitions partitions.csv --bin build/<app>.bin` |
| Mission completeness | `validate_mission` | `esp32wb mission validate missions/<file>.md` |

Use the [test plan template](assets/test_plan_template.md).

## Steps

### 1. Classify the change → layers

| Layer | Applies when | Tooling |
|-------|--------------|---------|
| 0. Static | Always | `esp32wb doctor`, compiler warnings (`-Werror` where possible), `idf.py clang-check` / cppcheck if available |
| 1. Repository logic | Python tooling, scripts, templates | `pytest` |
| 2. Host-side C logic | Pure logic: parsers, state machines, protocol encoders, math | Native build (CMake + Unity/CMock, or ESP-IDF `linux` target where the component supports it) |
| 3. On-target unit | Component behaviour needing the real chip or drivers | ESP-IDF Unity test app per component (`test_apps/`), run via `pytest-embedded` |
| 4. Target integration | End-to-end behaviour: boot, Wi-Fi, peripherals, OTA, power | `pytest-embedded` with `pytest-embedded-idf` + `pytest-embedded-serial-esp` |

Record the **target chip** for layers 3–4; a test passing on classic ESP32 says nothing
about ESP32-S3 or a RISC-V chip.

### 2. Enumerate cases per layer

- **Positive**: valid inputs, expected behaviour.
- **Negative**: invalid input, NULL handles, driver returning errors, missing NVS keys.
- **Boundary**: buffer sizes, max lengths, timeouts, ADC/PWM range ends, queue full.
- **Concurrency**: two tasks on shared state, ISR + task, dual-core contention (if the chip is dual-core).
- **Error / recovery**: bus NACK/timeout, Wi-Fi drop and reconnect, brownout, OTA failure + rollback.
- **Resource**: heap and stack high-water marks before/after; no leak over N iterations.

### 3. Pass/fail criteria and evidence

For each case: expected result, tolerance (timing/analog), and **evidence** — test
output, assertion count, serial log excerpt, measured value, or tool JSON.

### 4. Infrastructure

New test files, fixtures/mocks, test data, sdkconfig for tests
(`sdkconfig.ci.*`), and hardware setup (board, wiring, instruments).

### 5. Write the plan

Fill the template; store it with the mission (or in the mission's **Test Plan**
section for small changes).

## pytest conventions

- Mark hardware tests so CI can skip them:
  ```python
  @pytest.mark.hardware
  @pytest.mark.esp32s3        # pytest-embedded target marker, if used
  def test_wifi_reconnect(dut) -> None:
      dut.expect("Got IP", timeout=30)
  ```
- Keep host tests free of serial ports and network.
- Type hints, `pathlib.Path`, fixtures for shared setup (project standard).

## Rules

1. No contract without a test plan; tiny changes may use a 2–3 line plan.
2. Prefer extracting pure logic so it can be tested on the host.
3. Every hardware test is explicitly marked and lists required wiring.
4. Include negative tests — happy-path-only is insufficient for firmware.
5. "What was **not** tested and why" is a mandatory section.
