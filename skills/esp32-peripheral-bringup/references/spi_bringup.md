# Playbook: SPI Bring-Up

## Trigger

SPI peripheral fails to communicate with a slave device.

## Symptoms

- `spi_device_transmit()` returns `ESP_ERR_TIMEOUT`.
- Data read from slave is all `0xFF` or all `0x00`.
- Slave device does not respond.
- Data corruption — bits shifted or garbled.
- Intermittent communication failures at higher clock speeds.
- CS line behavior incorrect.

## Triage Steps

### Step 1: Verify Pin Selection

```c
spi_bus_config_t bus_cfg = {
    .mosi_io_num = 23,   // Master Out, Slave In
    .miso_io_num = 19,   // Master In, Slave Out
    .sclk_io_num = 18,   // Clock
    .quadwp_io_num = -1, // Not used
    .quadhd_io_num = -1, // Not used
};
```

**Check** (run `esp32wb pins .` for the configured target):
- [ ] MOSI, SCLK, CS are output-capable (not input-only on this chip).
- [ ] Not flash/PSRAM pins of **this** chip (SPI0/SPI1 are reserved for flash/PSRAM;
      use SPI2/SPI3 as general-purpose hosts, as available on the target).
- [ ] Not USB D-/D+ or UART0 console pins.
- [ ] Not shared with I2C or other peripherals (devices on the same SPI bus share
      MOSI/MISO/SCLK and need separate CS lines).
- [ ] CS not on a strapping pin unless the idle level is safe at reset.
- [ ] Wiring: MOSI→SDI/DIN, MISO→SDO/DOUT (not crossed like UART).

**IOMUX vs GPIO matrix**: each SPI host has dedicated IOMUX pins that allow the
highest clock; any other pin routes through the GPIO matrix, which lowers the maximum
usable frequency. IOMUX pin numbers and the limits differ per chip — look them up in the
SPI master driver docs for your target.

Classic ESP32 only (for reference):

| SPI Bus | MOSI | MISO | SCLK | CS | Notes |
|---------|------|------|------|----|-------|
| SPI2 (HSPI) | 13 | 12 | 14 | 15 | GPIO12 and 15 are strapping pins; 12–15 are JTAG |
| SPI3 (VSPI) | 23 | 19 | 18 | 5 | GPIO5 is a strapping pin |

### Step 2: Check SPI Mode (Clock Polarity and Phase)

SPI has 4 modes defined by CPOL and CPHA:

| Mode | CPOL | CPHA | Clock Idle | Data Sampled |
|------|------|------|-----------|-------------|
| 0 | 0 | 0 | Low | Rising edge |
| 1 | 0 | 1 | Low | Falling edge |
| 2 | 1 | 0 | High | Falling edge |
| 3 | 1 | 1 | High | Rising edge |

```c
spi_device_interface_config_t dev_cfg = {
    .mode = 0,  // Must match slave device datasheet
    // ...
};
```

**This is the #1 cause of SPI communication failures.** Check the slave device datasheet for the correct mode.

### Step 3: Check Clock Speed

```c
spi_device_interface_config_t dev_cfg = {
    .clock_speed_hz = 1000000,  // Start at 1 MHz for debugging
    // ...
};
```

**Debugging approach:**
1. Start at 1 MHz — if this works, the wiring is correct.
2. Gradually increase to target speed.
3. If failures occur at speed X, check:
   - Wire length (keep short for high speeds).
   - Signal integrity (use oscilloscope if available).
   - Slave device maximum clock specification.

Maximum speeds depend on the chip, IOMUX vs GPIO-matrix routing, full- vs half-duplex
and input delay of the slave — see the "Timing Considerations" section of the SPI
master docs for your target (`esp32wb docs search "SPI master timing" --target T`).

### Step 4: Check CS (Chip Select) Configuration

```c
spi_device_interface_config_t dev_cfg = {
    .spics_io_num = 5,     // Hardware CS management
    // OR
    .spics_io_num = -1,    // Manual CS management
    // ...
};
```

**Hardware CS issues:**
- Some devices need longer CS setup/hold than the default.
- Fix: Use `cs_ena_pretrans` and `cs_ena_posttrans` for timing control.

**Manual CS (recommended for problem devices):**
```c
gpio_set_level(CS_PIN, 0);  // Select
spi_device_transmit(handle, &transaction);
gpio_set_level(CS_PIN, 1);  // Deselect
```

### Step 5: Check Transaction Configuration

```c
spi_transaction_t trans = {
    .length = 8 * num_bytes,  // Length in BITS, not bytes!
    .tx_buffer = tx_data,
    .rx_buffer = rx_data,
};
```

**Common mistakes:**
- `.length` is in **bits**, not bytes. Multiply by 8.
- Using `.tx_data` (4-byte inline) when data is longer — use `.tx_buffer` for >4 bytes.
- Not setting `.rx_buffer` for read operations → data lost.
- Not setting `.flags = SPI_TRANS_USE_RXDATA` for small (≤ 4 byte) inline reads.

### Step 6: Check DMA Configuration

```c
// DMA channel allocation
spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
```

**DMA requirements:**
- DMA buffers must be in DMA-capable memory.
- Allocate with `heap_caps_malloc(size, MALLOC_CAP_DMA)`.
- Buffer alignment: at least 4-byte aligned; chips with cache-coherent PSRAM DMA may
  need cache-line alignment — check the driver docs (newer IDF versions provide a
  DMA-aware allocation helper).
- Small transfers (< 32 bytes): consider `SPI_DMA_DISABLED` to avoid DMA overhead.

**DMA errors:**
- `ESP_ERR_NO_MEM` during bus init → not enough DMA-capable memory.
- Data corruption with DMA → check buffer alignment.

### Step 7: Verify with Logic Analyzer

If software debugging doesn't resolve the issue:
1. Verify CS goes LOW during transaction.
2. Verify SCLK has the expected frequency and polarity.
3. Verify MOSI data matches what you're sending.
4. Verify MISO has response data from slave.
5. Check timing between CS assertion and first clock edge.

## Resolution Checklist

- [ ] Pin assignments verified (not reserved, not shared).
- [ ] SPI mode matches slave device datasheet.
- [ ] Communication works at 1 MHz.
- [ ] CS line is correctly managed (hardware or manual).
- [ ] Transaction length specified in bits.
- [ ] DMA buffers are properly aligned and in DMA-capable memory.
- [ ] Speed gradually increased to target from working 1 MHz baseline.

## Prevention

- Always start SPI bring-up at low speed (1 MHz).
- Choose SPI pins with `esp32wb pins` / `esp32wb chip` for the actual target; on classic
  ESP32, the VSPI (SPI3) IOMUX pins avoid most strapping conflicts.
- Default to SPI Mode 0 unless datasheet specifies otherwise.
- Use manual CS for devices with non-standard timing.
- Keep SPI bus wires short for high-speed operation.
- Document the SPI mode and speed in the pin map.
