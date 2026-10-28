# Playbook: I2C Bring-Up

## Trigger

I2C peripheral fails to communicate with a slave device.

## Symptoms

- Transfers return `ESP_ERR_TIMEOUT`, `ESP_FAIL` or `ESP_ERR_INVALID_STATE`.
- No ACK from the device; bus scan finds nothing.
- Data corruption or wrong values read.
- Bus hangs (SDA stuck LOW).

## Triage Steps

### Step 1: Verify Pin Selection

ESP-IDF has no fixed I2C pins — any output-capable GPIO works through the GPIO matrix,
so pins must be set explicitly. (SDA 21 / SCL 22 is only an Arduino convention on
classic ESP32 boards; it is meaningless on most other chips.)

**Check** (run `esp32wb pins .` for the configured target):
- [ ] SDA and SCL are output-capable (not input-only — classic ESP32 34–39, S2 GPIO46).
- [ ] Not flash/PSRAM pins, not USB D-/D+, not UART0 console pins of **this** chip.
- [ ] Not shared with another peripheral unless it is the same I2C bus.
- [ ] Pins match the physical wiring / board schematic (on-board sensors often fix the bus pins).

### Step 2: Check Pull-Up Resistors

I2C **requires** pull-up resistors on both SDA and SCL.

| Speed | Typical Pull-Up (3.3 V, short bus) |
|-------|-------------------|
| 100 kHz (Standard) | 4.7 kΩ |
| 400 kHz (Fast) | 2.2 kΩ |

Internal GPIO pull-ups (tens of kΩ) are **too weak** for reliable I2C; use them only
for a quick bench test. Many breakout boards already include pull-ups — too many in
parallel can also be a problem.

### Step 3: Configure the Driver

**ESP-IDF v5.2+ (new master driver, `driver/i2c_master.h`)**:
```c
i2c_master_bus_config_t bus_cfg = {
    .i2c_port = -1,                      // auto-select a free port
    .sda_io_num = I2C_SDA_GPIO,
    .scl_io_num = I2C_SCL_GPIO,
    .clk_source = I2C_CLK_SRC_DEFAULT,
    .glitch_ignore_cnt = 7,
    .flags.enable_internal_pullup = false, // external pull-ups fitted
};
i2c_master_bus_handle_t bus;
ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &bus));

i2c_device_config_t dev_cfg = {
    .dev_addr_length = I2C_ADDR_BIT_LEN_7,
    .device_address = 0x68,
    .scl_speed_hz = 100000,              // start slow
};
i2c_master_dev_handle_t dev;
ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &dev_cfg, &dev));
```

**Legacy driver (`driver/i2c.h`, ESP-IDF v4.x / deprecated in v5)**: `i2c_param_config()`
then `i2c_driver_install()`. Don't use both drivers in one application.

Confirm field names against the docs for your exact IDF version
(`esp32wb docs search "i2c_master_bus_config_t" --target T`).

**Common mistakes:**
- Port number beyond what the chip has (number of I2C controllers differs per target; some
  chips also have an LP I2C with different rules).
- Clock speed too high for bus capacitance / pull-up value.
- Missing power-up delay before the first transaction.

### Step 4: Verify Slave Address

- I2C addresses are 7-bit. Some datasheets show 8-bit (with R/W bit): `0xD0` → `0x68`.
- Address-select pins (A0/A1/SA0) change the address.

**Scan the bus (new driver):**
```c
for (uint16_t addr = 0x08; addr < 0x78; addr++) {
    if (i2c_master_probe(bus, addr, 50) == ESP_OK) {
        ESP_LOGI(TAG, "Found device at 0x%02X", addr);
    }
}
```

### Step 5: Read a Known Register

Read a WHO_AM_I / chip-ID register and compare with the datasheet value before doing
anything else. Use bounded timeouts on every transfer (`i2c_master_transmit_receive(dev,
..., timeout_ms)`).

### Step 6: Bus Recovery

If SDA is stuck LOW (slave mid-transfer after an MCU reset):
- New driver: `i2c_master_bus_reset(bus)` clocks SCL to release the bus.
- Manual: drive SCL as GPIO output and toggle ~9 clocks with SDA released, issue a STOP,
  then re-initialise the driver.

## Resolution Checklist

- [ ] Pins verified for the target chip and board.
- [ ] External pull-ups present with suitable value.
- [ ] Slave address confirmed (7-bit).
- [ ] Probe finds the device; ID register matches.
- [ ] Communication succeeds at 100 kHz, then at target speed.
- [ ] Timeouts, error handling and bus recovery implemented.

## Prevention

- Always start bring-up at 100 kHz.
- Always use external pull-ups.
- Document pin assignments in the project pin map.
- Protect a shared bus with the driver's thread-safety model or a mutex.
