pub mod api;
mod audio;
mod io;
mod processing;

use flutter_rust_bridge::frb;

#[frb(init)]
pub fn init_app() {
    flutter_rust_bridge::setup_default_user_utils();

    #[cfg(target_os = "android")]
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Debug)
            .with_tag("DeepFilterRust"),
    );

    #[cfg(not(target_os = "android"))]
    {
        // Simple logging for desktop
        let _ = env_logger::try_init();
    }
}
