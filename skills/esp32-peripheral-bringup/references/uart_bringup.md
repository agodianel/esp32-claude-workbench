# Playbook: UART Bring-Up

## Trigger

A UART peripheral (GPS, modem, RS-485 transceiver, another MCU) sends nothing, garbage,
or drops data.

## Symptoms

- `uart_read_bytes()` returns 0 / times out.
- Garbled characters (wrong baud or framing).
- Lost bytes at high rates (`UART_FIFO_OVF` / `UART_BUFFER_FULL` events).
- Console/logs disappear after UART init.

## Triage Steps

### Step 1: Pins and Port

- [ ] Pick a UART port that exists on the chip (UART count differs per target; some chips
      also have an LP UART). UART0 is normally the console.
- [ ] Do not reuse the chip's UART0 console TX/RX pins (`esp32wb chip <target>` lists them)
      unless the console is moved (`CONFIG_ESP_CONSOLE_*`) — e.g. to USB-Serial-JTAG on
      chips that have it.
- [ ] TX/RX crossed: ESP TX → device RX, ESP RX ← device TX; common ground.
- [ ] Voltage levels: 3.3 V logic only; RS-232 needs a level shifter; RS-485 needs a
      transceiver with DE/RE control.
- [ ] Pins checked with `esp32wb pins .` (no flash/PSRAM/USB pins, RX may use an input-only
      pin on chips that have them, TX may not).

### Step 2: Configuration

```c
const uart_port_t port = UART_NUM_1;
uart_config_t cfg = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity    = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    .source_clk = UART_SCLK_DEFAULT,
};
// RX buffer must be larger than the hardware FIFO (SOC_UART_FIFO_LEN)
ESP_ERROR_CHECK(uart_driver_install(port, 2048, 0, 20, &uart_queue, 0));
ESP_ERROR_CHECK(uart_param_config(port, &cfg));
ESP_ERROR_CHECK(uart_set_pin(port, TX_GPIO, RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
```

Match baud, data bits, parity, stop bits and flow control exactly with the device.

### Step 3: Loopback Test

Short TX to RX (or use `uart_set_loop_back()` where supported) and echo a known string.
If loopback works but the device doesn't, the problem is wiring, levels or device config.

### Step 4: Data Loss

- Use the event queue and handle `UART_FIFO_OVF` / `UART_BUFFER_FULL` by flushing
  (`uart_flush_input()`) and resynchronising.
- Increase the RX ring buffer; read in a dedicated task with adequate priority.
- Enable hardware flow control (RTS/CTS) for high baud rates.
- Use `uart_enable_pattern_det_baud_intr()` for line-based protocols (e.g. NMEA, AT).

### Step 5: RS-485

Use `uart_set_mode(port, UART_MODE_RS485_HALF_DUPLEX)` with the RTS pin driving DE/RE;
check transceiver termination and biasing.

## Resolution Checklist

- [ ] Correct port and pins for the chip; console untouched or moved deliberately.
- [ ] Framing matches on both sides.
- [ ] Loopback passes.
- [ ] Device responds; overflow handling in place.
- [ ] Timeouts on every read.
