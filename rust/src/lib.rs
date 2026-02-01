pub mod api;
mod audio;
mod io;
mod processing;

use flutter_rust_bridge::frb;
use std::sync::Once;

static INIT: Once = Once::new();

#[frb(init)]
pub fn init_app() {
    INIT.call_once(|| {
        #[cfg(target_os = "android")]
        {
            android_logger::init_once(
                android_logger::Config::default()
                    .with_max_level(log::LevelFilter::Trace)
                    .with_tag("DeepFilterRust"),
            );

            // Set panic hook to log panics
            std::panic::set_hook(Box::new(|panic_info| {
                let msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                let location = panic_info.location()
                    .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                    .unwrap_or_else(|| "unknown".to_string());

                log::error!("RUST PANIC: {} at {}", msg, location);
            }));
        }

        #[cfg(not(target_os = "android"))]
        {
            let _ = env_logger::try_init();
        }

        log::info!("DeepFilterRust initialized");
    });

    flutter_rust_bridge::setup_default_user_utils();
}
