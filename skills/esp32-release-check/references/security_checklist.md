# Security Readiness Checklist

> Security features are **chip- and silicon-revision-specific** and most are enforced
> by **one-time-programmable eFuses**. Confirm every item in the ESP-IDF
> "Security" guide for your exact target (`esp32wb docs search "<feature>" --target T`)
> before recommending it. Never burn eFuses as part of a review.

## 1. Secure Boot

- [ ] Decision recorded: enabled for production, or risk explicitly accepted.
- [ ] Scheme supported on this chip/revision (Secure Boot V2 is the current scheme;
      classic ESP32 supports V2 only from silicon revision v3.0 — older revisions only V1;
      signature algorithm (RSA / ECDSA) support differs per chip — check the docs).
- [ ] Signing keys generated and stored offline/HSM; key revocation plan documented.
- [ ] Bootloader size still fits before the partition table offset with secure boot enabled
      (secure boot grows the bootloader; may require moving `CONFIG_PARTITION_TABLE_OFFSET`).
- [ ] OTA images are signed in CI; unsigned image rejected in a test.

## 2. Flash Encryption

- [ ] Enabled for production in **Release** mode (Development mode allows re-flashing
      plaintext via serial and is not for shipped devices).
- [ ] Encrypted partitions flagged correctly (`encrypted` flag for data partitions that
      need it; app partitions are always encrypted when the feature is on).
- [ ] Manufacturing flow tested: first boot encrypts in place (can take time — do not cut power).
- [ ] Understand limits: number of times flash encryption can be toggled in Development
      mode is limited by eFuse bits.

## 3. NVS Encryption

- [ ] Enabled if NVS holds secrets (Wi-Fi credentials, tokens, keys).
- [ ] Key scheme chosen (flash-encryption-based `nvs_keys` partition, or HMAC-based on
      chips with the HMAC peripheral — check support for the target).

## 4. Debug and download interfaces

- [ ] JTAG: pad JTAG and USB-Serial-JTAG disabled via eFuse for production, or justified.
      (Enabling secure boot / flash encryption in Release mode disables JTAG on many
      chips by default — verify for the target.)
- [ ] ROM download mode: disabled or **Secure Download Mode** where supported.
- [ ] UART console: production log level; no interactive console exposing commands.

## 5. Application

- [ ] TLS everywhere; server certificates verified (`esp_crt_bundle_attach` or pinned CA);
      `CONFIG_ESP_TLS_INSECURE` / skip-verify **off**.
- [ ] No secrets in source, `sdkconfig.defaults`, logs, or core dumps (core dumps capture RAM).
- [ ] Device identity / provisioning flow does not leave default credentials.
- [ ] OTA server authentication and image signing (or secure boot) prevent arbitrary firmware.
- [ ] Anti-rollback secure version strategy if downgrade attacks matter.

## 6. Evidence

- `esp32wb sdkconfig` output for the release config (flags insecure options).
- Test logs: signed OTA accepted, unsigned/tampered OTA rejected, rollback works.
- `espefuse.py summary` from a production-configured sample unit (read-only command).
