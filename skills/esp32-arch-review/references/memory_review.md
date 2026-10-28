# Playbook: Memory Review

## Trigger

Suspected memory issues: heap exhaustion, memory leaks, stack overflows, or fragmentation.

## Symptoms

- `malloc()` or `heap_caps_malloc()` returns NULL.
- Free heap decreasing over time.
- Stack overflow crashes: "Stack canary watchpoint triggered" (Xtensa), "Stack protection
  fault" or a `Breakpoint` panic (RISC-V), or `LoadProhibited` / `IllegalInstruction` /
  access faults with a stack pointer near the task's stack limit.
- Random crashes after extended runtime.
- `MALLOC_FAILURE` or `heap corruption detected` in logs.

## Triage Steps

### Step 1: Check Current Memory Usage

Add runtime monitoring:
```c
// Snapshot of current heap
ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());
ESP_LOGI(TAG, "Min free heap: %lu bytes", esp_get_minimum_free_heap_size());

// Detailed heap info
heap_caps_print_heap_info(MALLOC_CAP_DEFAULT);

// Per-capability breakdown (sizes are size_t -> %u with a cast)
ESP_LOGI(TAG, "Free internal: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
ESP_LOGI(TAG, "Free DMA: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_DMA));
ESP_LOGI(TAG, "Free PSRAM: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));  // 0 if no PSRAM
```

### Step 2: Detect Memory Leaks

**Periodic monitoring:**
```c
// Create a monitoring task
void memory_monitor_task(void *pvParameters) {
    while (1) {
        ESP_LOGI("MEM", "Free: %lu  Min: %lu",
                 esp_get_free_heap_size(),
                 esp_get_minimum_free_heap_size());
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
```

If free heap steadily decreases, there is a leak.

**Enable heap tracing:**
```
CONFIG_HEAP_TRACING_STANDALONE=y
```
```c
#include "esp_heap_trace.h"

#define TRACE_RECORD_COUNT 200
static heap_trace_record_t trace_records[TRACE_RECORD_COUNT];

// Before the suspected leaking code:
heap_trace_init_standalone(trace_records, TRACE_RECORD_COUNT);
heap_trace_start(HEAP_TRACE_LEAKS);

// After the suspected leaking code:
heap_trace_stop();
heap_trace_dump();
```

### Step 3: Detect Heap Corruption

Enable comprehensive heap poisoning:
```
CONFIG_HEAP_POISONING_COMPREHENSIVE=y
```

This fills freed memory with a pattern and checks it on allocation. If corruption is detected, you get an immediate assert with the corruption site.

**Also enable:**
```
CONFIG_HEAP_TASK_TRACKING=y  # Track which task allocated each block
```

Comprehensive poisoning costs RAM and CPU time — disable it in release builds
(`esp32wb sdkconfig` flags debug options left on).

### Step 4: Check Stack Usage

```c
// Check remaining stack for current task.
// ESP-IDF FreeRTOS measures stacks in BYTES (unlike vanilla FreeRTOS, which uses words),
// so the high-water mark is already in bytes.
UBaseType_t stack_remaining = uxTaskGetStackHighWaterMark(NULL);
ESP_LOGI(TAG, "Stack high water mark: %u bytes free", (unsigned)stack_remaining);

// Check all tasks (requires CONFIG_FREERTOS_USE_TRACE_FACILITY=y and
// CONFIG_FREERTOS_USE_STATS_FORMATTING_FUNCTIONS=y)
char *task_list = malloc(2048);
if (task_list) {
    vTaskList(task_list);
    ESP_LOGI(TAG, "Task          State  Prio  Stack  Num\n%s", task_list);
    free(task_list);
}
```

Static check without hardware: `esp32wb stacks .` (MCP `stack_check`) flags
undersized `xTaskCreate*` stack arguments.

**Stack sizing guidelines:**

Starting points only — confirm with high-water marks on the real target:

| Task Type | Starting Stack (bytes) |
|-----------|------------------------|
| Simple processing | 2048 bytes |
| With string operations | 3072 bytes |
| Wi-Fi / TCP/IP | 4096 bytes |
| TLS / HTTPS | 8192 bytes |
| JSON parsing (large) | 8192+ bytes |

Enable stack checking:
```
CONFIG_FREERTOS_WATCHPOINT_END_OF_STACK=y
CONFIG_COMPILER_STACK_CHECK=y
```

### Step 5: Review Allocation Patterns

**Common memory mistakes:**

```c
// BAD: Large stack allocation
void process() {
    char buffer[4096];  // Stack allocation — risky!
}

// GOOD: Heap allocation
void process() {
    char *buffer = malloc(4096);
    if (!buffer) { ESP_LOGE(TAG, "alloc failed"); return; }
    // ... use buffer ...
    free(buffer);
}

// BAD: Leak on error path
void init() {
    char *a = malloc(100);
    char *b = malloc(200);
    if (!b) return;  // Leaked 'a'!
    // ...
}

// GOOD: Clean error handling
void init() {
    char *a = malloc(100);
    if (!a) return;
    char *b = malloc(200);
    if (!b) { free(a); return; }
    // ...
}
```

### Step 6: Check Fragmentation

```c
// Largest free block vs total free
size_t free_bytes = heap_caps_get_free_size(MALLOC_CAP_8BIT);
size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
ESP_LOGI(TAG, "Free: %u  Largest block: %u  Fragmentation: %.1f%%",
         (unsigned)free_bytes, (unsigned)largest,
         (1.0 - (double)largest / free_bytes) * 100);
```

If total free is large but largest block is small, heap is fragmented.

**Mitigation:**
- Avoid frequent alloc/free of different sizes.
- Pre-allocate buffers at startup.
- Use memory pools for fixed-size objects.

## Resolution Checklist

- [ ] Current memory usage measured.
- [ ] Leak presence confirmed or ruled out.
- [ ] Heap corruption checked with poisoning.
- [ ] Stack high water marks reviewed for all tasks.
- [ ] Fragmentation level assessed.
- [ ] Fix applied and verified under sustained load.

## Prevention

- Monitor free heap in production builds.
- Set alerts when free heap drops below a threshold.
- Size task stacks generously during development, tune later.
- Use heap tracing during development.
- Pre-allocate where possible to avoid fragmentation.
- Always free memory on error paths.
