extern crate deepfilter_audio;

mod frb_generated;
pub mod api;

// Re-export init_app for flutter_rust_bridge
pub use deepfilter_audio::init_app;
