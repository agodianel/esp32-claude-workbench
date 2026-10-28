# Playbook: ADC Bring-Up

## Trigger

ADC readings are wrong, noisy, saturated, or fail — especially with Wi-Fi active.

## Symptoms

- Reads stuck at 0 or full scale.
- Values differ from a multimeter by more than expected.
- `adc_oneshot_read()` returns `ESP_ERR_TIMEOUT` / error while Wi-Fi is on.
- Very noisy readings.

## Triage Steps

### Step 1: Pin → ADC unit/channel for THIS chip

ADC channel maps are completely different per target. Get them from
`esp32wb chip <target>` (ADC1 / ADC2 lines) or use `adc_oneshot_io_to_channel()` in
code. Never reuse classic-ESP32 channel numbers on another chip.

### Step 2: ADC2 and Wi-Fi (chip-specific)

| Chip | ADC2 behaviour |
|------|----------------|
| Classic ESP32 | ADC2 cannot be used while Wi-Fi is used → use ADC1 |
| ESP32-S2 / ESP32-S3 | Shared with Wi-Fi; `adc_oneshot_read()` arbitrates and can fail while Wi-Fi uses it; S3 ADC2 continuous (DMA) mode not supported |
| ESP32-C3 | ADC2 oneshot no longer supported in current ESP-IDF (hardware erratum) → use ADC1 |
| Others | Check `esp32wb chip <target>`; several have no ADC2 |

### Step 3: Use the v5 driver (`esp_adc` component)

```c
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

adc_oneshot_unit_handle_t adc;
adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = ADC_UNIT_1 };
ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &adc));

adc_oneshot_chan_cfg_t ch_cfg = {
    .atten = ADC_ATTEN_DB_12,        // DB_11 in older IDF; range depends on chip
    .bitwidth = ADC_BITWIDTH_DEFAULT,
};
ESP_ERROR_CHECK(adc_oneshot_config_channel(adc, ADC_CHANNEL_X, &ch_cfg));

int raw;
esp_err_t err = adc_oneshot_read(adc, ADC_CHANNEL_X, &raw);  // check err (ADC2 arbitration!)
```

Legacy `driver/adc.h` and `esp_adc_cal` are deprecated in v5.

### Step 4: Calibrate

Calibration schemes are chip-dependent. Select at compile time:
```c
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t c = { .unit_id = ADC_UNIT_1, .chan = ADC_CHANNEL_X,
                                          .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT };
    adc_cali_create_scheme_curve_fitting(&c, &cali);
#elif ADC_CALI_SCHEME_LINE_FITTING_SUPPORTED
    adc_cali_line_fitting_config_t c = { .unit_id = ADC_UNIT_1,
                                         .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT };
    adc_cali_create_scheme_line_fitting(&c, &cali);
#endif
adc_cali_raw_to_voltage(cali, raw, &mv);
```
Handle `ESP_ERR_NOT_SUPPORTED` (eFuse calibration data missing on some early samples).

### Step 5: Input Range and Signal

- Attenuation sets the measurable range; the usable range is chip-specific and
  non-linear near the ends — check the ADC docs for your target.
- Never exceed the pin's absolute maximum voltage; use a divider for higher voltages
  and account for its source impedance.
- Add a small capacitor at the pin for high-impedance sources; average multiple samples.
- Input-only pins on classic ESP32 (34–39) are fine for ADC.

## Resolution Checklist

- [ ] Channel mapping taken from the chip's data, not memory.
- [ ] ADC1 used if Wi-Fi is active (or ADC2 errors handled on S2/S3).
- [ ] `esp_adc` oneshot/continuous driver used; return codes checked.
- [ ] Calibration scheme selected by capability macro; readings in mV.
- [ ] Verified against a reference voltage at low/mid/high points.
