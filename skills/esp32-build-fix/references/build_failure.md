# Playbook: Build Failure Triage

## Trigger

ESP-IDF project fails to compile (`idf.py build` or `cmake --build`).

## Symptoms

- Build errors in terminal output.
- Linker errors.
- Component registration failures.
- Missing header files.

> Fastest first step: `idf.py build 2>&1 | tee build.log` then
> `esp32wb build-log build.log` (MCP `build_log_triage`) to classify the first error.

## Triage Steps

### Step 1: Identify Error Type

| Error Pattern | Category | Likely Cause |
|--------------|----------|-------------|
| `fatal error: xyz.h: No such file` | Include path | Missing component dependency or wrong include path |
| `undefined reference to` | Linker | Missing source file, missing component REQUIRES |
| `error: unknown type name` | Type | Missing include or wrong target configuration |
| `error: 'CONFIG_XYZ' undeclared` | Kconfig | Missing Kconfig option or sdkconfig mismatch |
| `CMake Error: Cannot find component` | CMake | Component not registered or wrong directory |
| `multiple definition of` | Linker | Duplicate symbol — same function defined twice |
| `error: static assertion failed` | Config | ESP-IDF version or target mismatch |
| `region 'iram0_0_seg' overflowed` / `dram0_0_seg` | Memory | Too much IRAM code / static data for this chip |
| `app partition is too small for binary` | Partition | App outgrew its partition |
| `implicit declaration of function` after upgrade | API | Function renamed/removed — see migration guide |
| `Failed to resolve component` | Component manager | Dependency missing or version constraint unsatisfiable |

### Step 2: Check Target Configuration

```bash
# What is configured?
grep CONFIG_IDF_TARGET= sdkconfig

# Switch target (WARNING: regenerates sdkconfig — back it up first;
# settings survive only if they are in sdkconfig.defaults[.<target>])
cp sdkconfig sdkconfig.bak
idf.py set-target esp32s3   # esp32 esp32s2 esp32s3 esp32c2 esp32c3 esp32c5 esp32c6 esp32h2 esp32p4
idf.py build
```

Per-target defaults belong in `sdkconfig.defaults.<target>` (e.g.
`sdkconfig.defaults.esp32s3`), which ESP-IDF applies automatically.

### Step 3: Check Component Dependencies

In each component's `CMakeLists.txt`:
```cmake
idf_component_register(
    SRCS "source.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_wifi       # Public dependencies
    PRIV_REQUIRES esp_timer nvs_flash  # Private dependencies
)
```

Common missing dependencies:
- `driver` — umbrella for GPIO, I2C, SPI, UART, LEDC (recent IDF v5.x splits these into
  `esp_driver_gpio`, `esp_driver_i2c`, `esp_driver_spi`, `esp_driver_uart`,
  `esp_driver_ledc`, …; `esp_adc` for ADC)
- `esp_wifi` — for Wi-Fi features
- `esp_event` — for event loop
- `nvs_flash` — for NVS storage
- `esp_http_client` — for HTTP
- `mqtt` — for MQTT

### Step 4: Check sdkconfig

```bash
# Review the active config
esp32wb sdkconfig sdkconfig

# Regenerate from defaults (DESTRUCTIVE for un-saved menuconfig changes — back up first)
cp sdkconfig sdkconfig.bak
rm sdkconfig
idf.py build
diff sdkconfig.bak sdkconfig   # anything important that vanished belongs in sdkconfig.defaults
```

Common sdkconfig issues:
- Wrong partition table for the flash size.
- Feature not enabled (e.g., BLE, OTA, HTTPS).
- Wrong CPU frequency setting.
- Incompatible options for the target chip.

### Step 5: Clean Build

```bash
# Standard clean (removes build/ contents; keeps sdkconfig)
idf.py fullclean
idf.py build
```

If `managed_components/` is suspect, delete it and `dependencies.lock` only with the
user's agreement — it changes resolved dependency versions.

### Step 6: Check ESP-IDF Version

```bash
# Check version
idf.py --version

# Check the ESP-IDF "Migration Guides" for every major/minor version you crossed.
# Well-known v4.x -> v5.x changes:
# - ADC: legacy driver/adc.h -> esp_adc (adc_oneshot / adc_continuous / adc_cali)
# - I2C: new driver/i2c_master.h API (legacy driver/i2c.h deprecated, do not mix both)
# - Timer Group -> gptimer, RMT/I2S/PCNT/MCPWM reworked
# - Log format specifiers: uint32_t is unsigned long on v5 -> use PRIu32 / %lu
# - FreeRTOS: portTICK_RATE_MS removed -> portTICK_PERIOD_MS / pdMS_TO_TICKS
```
Search with `esp32wb docs search "<symbol>"` to confirm the replacement API.

## Resolution Checklist

- [ ] Error type identified and categorized.
- [ ] Root cause found.
- [ ] Fix applied.
- [ ] Clean build succeeds.
- [ ] No new warnings introduced.

## Prevention

- Always declare component dependencies explicitly in `CMakeLists.txt`.
- Use `sdkconfig.defaults` to track all non-default configuration.
- Pin the ESP-IDF version in your project documentation.
- Run clean builds in CI.
