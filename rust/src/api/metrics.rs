use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use flutter_rust_bridge::frb;

/// CPU times from /proc/stat
static PREV_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREV_IDLE: AtomicU64 = AtomicU64::new(0);

/// Track if NNAPI was successfully registered (set by deepfilter.rs)
pub static NNAPI_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Set NNAPI registration status
pub fn set_nnapi_registered(registered: bool) {
    NNAPI_REGISTERED.store(registered, Ordering::SeqCst);
    log::info!("NNAPI registered status set to: {}", registered);
}

/// System metrics for UI display
#[frb]
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub nnapi_available: bool,
}

/// Get current system metrics (CPU/GPU usage)
#[frb]
pub fn get_system_metrics() -> SystemMetrics {
    let cpu_usage = get_cpu_usage();
    let gpu_usage = get_gpu_usage();
    let nnapi_available = check_nnapi_available();

    SystemMetrics {
        cpu_usage_percent: cpu_usage,
        gpu_usage_percent: gpu_usage,
        nnapi_available,
    }
}

/// Calculate CPU usage from /proc/stat
fn get_cpu_usage() -> f32 {
    // Works on both Android and Linux
    if let Ok(stat) = fs::read_to_string("/proc/stat") {
        if let Some(cpu_line) = stat.lines().next() {
            let parts: Vec<&str> = cpu_line.split_whitespace().collect();
            if parts.len() >= 5 && parts[0] == "cpu" {
                // cpu user nice system idle iowait irq softirq
                let user: u64 = parts[1].parse().unwrap_or(0);
                let nice: u64 = parts[2].parse().unwrap_or(0);
                let system: u64 = parts[3].parse().unwrap_or(0);
                let idle: u64 = parts[4].parse().unwrap_or(0);
                let iowait: u64 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
                let irq: u64 = parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
                let softirq: u64 = parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);

                let total = user + nice + system + idle + iowait + irq + softirq;
                let idle_total = idle + iowait;

                let prev_total = PREV_TOTAL.swap(total, Ordering::SeqCst);
                let prev_idle = PREV_IDLE.swap(idle_total, Ordering::SeqCst);

                if prev_total > 0 {
                    let total_diff = total.saturating_sub(prev_total);
                    let idle_diff = idle_total.saturating_sub(prev_idle);

                    if total_diff > 0 {
                        let usage = 100.0 * (1.0 - (idle_diff as f32 / total_diff as f32));
                        return usage.clamp(0.0, 100.0);
                    }
                } else {
                    // First read - return 0 instead of -1, next read will have delta
                    return 0.0;
                }
            }
        }
    } else {
        log::warn!("Failed to read /proc/stat for CPU usage");
    }

    -1.0 // Unsupported
}

/// Try to get GPU usage (device-specific)
fn get_gpu_usage() -> f32 {
    #[cfg(target_os = "android")]
    {
        // Try Qualcomm Adreno
        if let Ok(content) = fs::read_to_string("/sys/class/kgsl/kgsl-3d0/gpubusy") {
            let parts: Vec<&str> = content.trim().split_whitespace().collect();
            if parts.len() >= 2 {
                let busy: f32 = parts[0].parse().unwrap_or(0.0);
                let total: f32 = parts[1].parse().unwrap_or(1.0);
                if total > 0.0 {
                    return (busy / total * 100.0).clamp(0.0, 100.0);
                }
            }
        }

        // Try Qualcomm alternative path
        if let Ok(content) = fs::read_to_string("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage") {
            if let Ok(pct) = content.trim().parse::<f32>() {
                return pct.clamp(0.0, 100.0);
            }
        }

        // Try Mali GPU
        for path in &[
            "/sys/devices/platform/mali.0/utilization",
            "/sys/devices/platform/13000000.mali/utilization",
            "/sys/devices/platform/18500000.mali/utilization",
            "/sys/class/misc/mali0/device/utilization",
        ] {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(pct) = content.trim().parse::<f32>() {
                    return pct.clamp(0.0, 100.0);
                }
            }
        }

        // Try reading GPU frequency scaling (indirect measure)
        if let Ok(content) = fs::read_to_string("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq") {
            if let Ok(cur) = content.trim().parse::<f64>() {
                if let Ok(max_content) = fs::read_to_string("/sys/class/kgsl/kgsl-3d0/devfreq/max_freq") {
                    if let Ok(max) = max_content.trim().parse::<f64>() {
                        if max > 0.0 {
                            return ((cur / max) * 100.0).clamp(0.0, 100.0) as f32;
                        }
                    }
                }
            }
        }
    }

    -1.0 // Unsupported or not available
}

/// Check if NNAPI was successfully registered
fn check_nnapi_available() -> bool {
    // Return the actual registration status from model loading
    NNAPI_REGISTERED.load(Ordering::SeqCst)
}
