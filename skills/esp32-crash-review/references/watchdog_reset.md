# Playbook: Watchdog Reset Review

## Trigger

A watchdog message appears in the serial log, or the device resets with a watchdog
reset reason.

## Symptoms

- `Task watchdog got triggered. The following tasks did not reset the watchdog in time:`
- `Guru Meditation Error: Core N panic'ed (Interrupt wdt timeout on CPUN)`
- `esp_reset_reason()` returns `ESP_RST_TASK_WDT`, `ESP_RST_INT_WDT` or `ESP_RST_WDT`.
- Periodic resets during specific operations (flash writes, long computations, Wi-Fi bursts).

## Triage Steps

### Step 1: Identify the Watchdog by its Message (not by the ROM reset code)

ROM reset codes and names differ per chip, and a watchdog that goes through the panic
handler usually ends in a *software* CPU reset. Identify the watchdog from the text:

| Message | Watchdog | Default behaviour (ESP-IDF) |
|---------|----------|-----------------------------|
| `Task watchdog got triggered` | Task WDT (TWDT) — watches IDLE tasks and subscribed tasks | Prints the offending tasks + backtrace and **keeps running**; panics/resets only if `CONFIG_ESP_TASK_WDT_PANIC=y` |
| `Interrupt wdt timeout on CPU0/CPU1` | Interrupt WDT (IWDT) — ISRs / interrupts disabled too long | Invokes the panic handler (Guru Meditation), then reboots per `CONFIG_ESP_SYSTEM_PANIC` |
| ROM reports a hardware watchdog system reset with no panic text | Second stage of a WDT fired because the panic handler itself hung, or the RTC WDT fired during boot | Hard reset |

Timeouts are configurable: `CONFIG_ESP_TASK_WDT_TIMEOUT_S`, `CONFIG_ESP_INT_WDT_TIMEOUT_MS`
(check your `sdkconfig` for the actual values; the IWDT default is larger when PSRAM
is enabled).

**Task WDT**: the IDLE task on a core could not run for the timeout → some task is
hogging that core without blocking.

**Interrupt WDT**: interrupt processing took too long or interrupts were disabled
too long (long critical sections, slow ISRs, flash operations with ISRs not in IRAM).

### Step 2: Identify the Blocking Task

The watchdog message shows which task was blocked:

```
E (12345) task_wdt: Task watchdog got triggered. The following tasks did not reset the watchdog in time:
E (12345) task_wdt:  - IDLE0 (CPU 0)
E (12345) task_wdt: Tasks currently running:
E (12345) task_wdt: CPU 0: my_blocking_task
E (12345) task_wdt: CPU 1: IDLE1
```

This tells you `my_blocking_task` is hogging CPU 0. Single-core chips (ESP32-S2,
C-series, H2) only show `IDLE` / CPU 0.

### Step 3: Find the Blocking Code

Common causes:

**Busy-wait loops:**
```c
// BAD: Blocks forever without yielding
while (!flag_is_set) {
    // No delay, no yield
}

// GOOD: Yield or delay
while (!flag_is_set) {
    vTaskDelay(pdMS_TO_TICKS(10));
}
```

**Long computations without yielding:**
```c
// BAD: Long loop without yield
for (int i = 0; i < 1000000; i++) {
    do_heavy_computation(i);
}

// GOOD: Yield periodically
for (int i = 0; i < 1000000; i++) {
    do_heavy_computation(i);
    if (i % 1000 == 0) {
        vTaskDelay(1);  // Yield to other tasks
    }
}
```

**Disabled interrupts for too long:**
```c
// BAD: Long critical section
taskENTER_CRITICAL(&spinlock);
process_large_buffer(buf, 10000);  // Takes too long
taskEXIT_CRITICAL(&spinlock);

// GOOD: Short critical sections
for (int i = 0; i < 100; i++) {
    taskENTER_CRITICAL(&spinlock);
    process_chunk(buf + i * 100, 100);
    taskEXIT_CRITICAL(&spinlock);
}
```

**Infinite loops in ISR:**
```c
// BAD: Complex logic in ISR
void IRAM_ATTR my_isr(void *arg) {
    while (data_available()) {
        process_data();  // Could take too long
    }
}

// GOOD: Defer work to task
void IRAM_ATTR my_isr(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xTaskNotifyFromISR(worker_task, 0, eNoAction, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

### Step 4: Check Task Priorities

```c
// A high-priority task that never blocks will starve lower priorities
// including the IDLE task (priority 0)
xTaskCreate(my_task, "my_task", 4096, NULL, configMAX_PRIORITIES - 1, NULL);
// If my_task never calls vTaskDelay or blocks on a queue/semaphore,
// the IDLE task never runs → watchdog triggers
```

### Step 5: Feed or Reconfigure the Watchdog

For legitimately long operations:
```c
#include "esp_task_wdt.h"

// Option 1: Feed the watchdog
esp_task_wdt_reset();

// Option 2: Add current task to WDT and manage manually
esp_task_wdt_add(NULL);
// ... long work with periodic esp_task_wdt_reset() ...
esp_task_wdt_delete(NULL);

// Option 3: Increase timeout (in menuconfig)
// CONFIG_ESP_TASK_WDT_TIMEOUT_S=30
```

## Resolution Checklist

- [ ] Watchdog type identified from the message (Task vs Interrupt vs hardware stage-2).
- [ ] Blocking task identified from WDT message.
- [ ] Root cause found (busy-wait, long compute, ISR abuse, priority issue).
- [ ] Fix applied (added yields, deferred ISR work, reduced critical sections).
- [ ] System runs stable under load after fix.

## Prevention

- Never write busy-wait loops without a yield or delay.
- Keep ISRs short — defer complex work to tasks.
- Keep critical sections under 1ms.
- Use appropriate task priorities and always include blocking points.
- Periodically feed the watchdog in legitimately long operations.
