# Playbook: LEDC (PWM) Bring-Up

## Trigger

PWM output (LED dimming, backlight, buzzer, servo) doesn't appear, has the wrong
frequency, or `ledc_timer_config()` fails.

## Symptoms

- `ledc_timer_config()` returns `ESP_FAIL` / logs that the frequency/resolution can't be achieved.
- Output stuck high/low.
- Flicker or glitches when changing duty.

## Triage Steps

### Step 1: Pin and Channel

- [ ] Output-capable pin for the target (`esp32wb pins .`); not a strapping pin whose PWM
      idle level could break boot, not USB/console pins.
- [ ] Channel/timer numbers exist on this chip (LEDC channel and timer counts differ per target).

### Step 2: Speed Mode

- `LEDC_HIGH_SPEED_MODE` exists **only on classic ESP32**.
- All other chips: `LEDC_LOW_SPEED_MODE` only (use `ledc_update_duty()` after setting duty).

### Step 3: Frequency vs Resolution

The timer must be able to produce `freq × 2^duty_resolution` from its clock source.
Higher frequency → lower achievable resolution.

```c
ledc_timer_config_t timer = {
    .speed_mode      = LEDC_LOW_SPEED_MODE,
    .timer_num       = LEDC_TIMER_0,
    .duty_resolution = LEDC_TIMER_10_BIT,
    .freq_hz         = 5000,
    .clk_cfg         = LEDC_AUTO_CLK,
};
ESP_ERROR_CHECK(ledc_timer_config(&timer));

ledc_channel_config_t ch = {
    .speed_mode = LEDC_LOW_SPEED_MODE,
    .channel    = LEDC_CHANNEL_0,
    .timer_sel  = LEDC_TIMER_0,
    .gpio_num   = PWM_GPIO,
    .duty       = 0,
    .hpoint     = 0,
};
ESP_ERROR_CHECK(ledc_channel_config(&ch));

ESP_ERROR_CHECK(ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, 512));
ESP_ERROR_CHECK(ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0));
```

If config fails, lower the resolution or the frequency. Clock sources available differ
per chip — check the LEDC docs for your target.

### Step 4: Duty Changes and Fades

- Duty range is `0 … 2^resolution` (full on is `2^resolution` on newer IDF; check docs).
- For fades: `ledc_fade_func_install(0)` once, then `ledc_set_fade_with_time()` +
  `ledc_fade_start()`. Don't mix fade and direct duty updates on the same channel concurrently.
- Duty updates from multiple tasks → protect with a mutex or use the thread-safe
  `ledc_set_duty_and_update()`.

### Step 5: Light Sleep

Some clock sources stop in light sleep; if PWM must keep running, choose a clock source
that stays on in sleep (chip-dependent).

### Step 6: Servos and Buzzers

- Servos: 50 Hz with ≥ 14-bit resolution for fine pulse-width control (if the clock allows).
- Passive buzzers: frequency = tone; 50 % duty; set duty 0 to silence.

## Resolution Checklist

- [ ] Correct speed mode for the chip.
- [ ] Frequency/resolution combination accepted by `ledc_timer_config()`.
- [ ] Waveform verified on a scope / logic analyzer or by visible LED behaviour.
- [ ] Return codes checked; concurrent duty updates protected.
