# Playbook: OTA Update Debugging

## Trigger

Over-the-air firmware update fails — download errors, verification failures, boot loops after update, or rollback problems.

## Symptoms

- OTA download starts but never completes.
- `esp_ota_end()` returns an error.
- Device boot-loops after OTA update.
- Device rolls back to old firmware unexpectedly.
- `ESP_ERR_OTA_VALIDATE_FAILED` in logs.
- Firmware version does not change after OTA.

## Triage Steps

### Step 1: Verify Partition Table

OTA requires at least two OTA app partitions (`ota_0`, `ota_1`) and an `otadata` partition.

```bash
# Print the partition table the build uses
idf.py partition-table

# Validate layout, alignment, flash fit and app size in one go
esp32wb partitions partitions.csv --sdkconfig sdkconfig --bin build/<app>.bin
```

Example OTA layout for 4 MB flash (adjust sizes to your app and flash):
```
# Name,   Type, SubType, Offset,  Size
nvs,      data, nvs,     0x9000,  0x4000
otadata,  data, ota,     0xd000,  0x2000
phy_init, data, phy,     0xf000,  0x1000
ota_0,    app,  ota_0,   0x10000, 0x180000
ota_1,    app,  ota_1,   0x190000,0x180000
```

**Common mistakes:**
- Missing `otadata` partition → OTA boot selection fails.
- Only one `ota_x` partition → no space for new firmware.
- Partitions too small → firmware doesn't fit.
- Using `factory` instead of `ota_0` → OTA has nowhere to write.

### Step 2: Check Firmware Size vs Partition Size

```bash
idf.py size                                   # app size summary
esp32wb partitions partitions.csv --bin build/<app>.bin   # fails if any app slot is too small
```

If the image is larger than the OTA slot, `esp_ota_write()` / `esp_https_ota()` fails
(e.g. `ESP_ERR_OTA_VALIDATE_FAILED` or a size error) and the device keeps the old app.
Gate this in CI so it never reaches the field.

### Step 3: Check OTA Process Code

```c
// Correct OTA sequence:
esp_http_client_config_t config = { .url = firmware_url, .cert_pem = ca_cert };
esp_https_ota_config_t ota_config = { .http_config = &config };

esp_err_t ret = esp_https_ota(&ota_config);
if (ret == ESP_OK) {
    ESP_LOGI(TAG, "OTA succeeded, restarting...");
    esp_restart();
} else {
    ESP_LOGE(TAG, "OTA failed: %s", esp_err_to_name(ret));
}
```

**Common OTA errors:**

| Error | Cause |
|-------|-------|
| `ESP_ERR_OTA_VALIDATE_FAILED` | Image header invalid, magic byte wrong, or image truncated |
| `ESP_ERR_NO_MEM` | Not enough heap for HTTP client + OTA buffer |
| `ESP_ERR_INVALID_SIZE` | Size problem writing the image (e.g. larger than the slot) |
| `ESP_ERR_OTA_ROLLBACK_INVALID_STATE` | Rollback state machine in wrong state |
| `ESP_ERR_HTTP_CONNECT` | Can't reach the update server |

### Step 4: Check TLS/HTTPS Configuration

Most OTA failures are actually HTTPS failures:

- [ ] CA certificate is correct and not expired.
- [ ] Server hostname matches certificate CN/SAN.
- [ ] `CONFIG_MBEDTLS_CERTIFICATE_BUNDLE=y` if using public CAs.
- [ ] `CONFIG_ESP_TLS_SKIP_SERVER_CERT_VERIFY` is NOT set to `y` in production.
- [ ] Enough free internal heap for the TLS handshake (measure; tens of KB is typical).

### Step 5: Check Anti-Rollback (if enabled)

If using secure version rollback:
```
CONFIG_BOOTLOADER_APP_ANTI_ROLLBACK=y
CONFIG_BOOTLOADER_APP_SECURE_VERSION=N
```

- New firmware must have `secure_version >= current_secure_version`.
- Check with: `esp_efuse_read_field_blob(ESP_EFUSE_SECURE_VERSION, ...)`.
- Once the secure version eFuse is burned, it cannot be reversed.

### Step 6: Debug Boot Loop After OTA

If device boot-loops after OTA update:

1. **Check app validity**: the new image may be corrupt or built for another target.
   The bootloader/OTA code reports a chip-ID mismatch in that case — rebuild with the
   correct `idf.py set-target`. `esp_https_ota` normally rejects such images before
   switching the boot partition.

2. **Check rollback** (only when `CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE=y`):
   ```c
   // In the new firmware, after its self-test passes:
   esp_ota_mark_app_valid_cancel_rollback();
   // or, if the self-test fails:
   esp_ota_mark_app_invalid_rollback_and_reboot();
   ```
   With rollback enabled, a new image boots in the *pending verify* state. If the
   device reboots before the app marks itself valid, the bootloader marks it invalid
   and boots the previous app. Without rollback enabled, these calls are not needed
   and a crashing new image keeps crash-looping.

3. **Crash loop after OTA**: decode the panic with the new image's ELF
   (`esp32wb crash <log> --elf <new app.elf>`) — keep ELFs for every released version.

### Step 7: Verify Version After OTA

```c
const esp_app_desc_t *app_desc = esp_app_get_description();
ESP_LOGI(TAG, "Running firmware: %s, version: %s",
         app_desc->project_name, app_desc->version);
```

## Resolution Checklist

- [ ] Partition table has `ota_0`, `ota_1`, and `otadata`.
- [ ] Firmware binary fits within OTA partition size.
- [ ] HTTPS certificate chain is valid.
- [ ] Free heap during OTA measured with margin.
- [ ] If rollback is enabled: `esp_ota_mark_app_valid_cancel_rollback()` called after self-test.
- [ ] Firmware version reported correctly after update.
- [ ] Rollback works when new firmware fails.

## Prevention

- Always validate firmware size against partition size in CI.
- Use `esp_https_ota()` instead of manual OTA for simpler error handling.
- Enable app rollback for fleet devices and mark the app valid only after a real self-test.
- Test the full OTA cycle (download → flash → boot → validate → rollback) before release.
- Log the firmware version at boot to confirm updates.
- Measure minimum free heap during a full OTA and keep a margin.
