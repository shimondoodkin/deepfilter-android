// Wrapper API for flutter_rust_bridge
// Re-exports deepfilter_audio functions with #[frb] annotations

use flutter_rust_bridge::frb;

/// Recording status for UI updates
pub struct RecordingStatus {
    pub is_recording: bool,
    pub is_playing: bool,
    pub duration_ms: u64,
    pub stream_a_enabled: bool,
    pub stream_b_enabled: bool,
    pub error: Option<String>,
}

impl From<deepfilter_audio::api::RecordingStatus> for RecordingStatus {
    fn from(s: deepfilter_audio::api::RecordingStatus) -> Self {
        Self {
            is_recording: s.is_recording,
            is_playing: s.is_playing,
            duration_ms: s.duration_ms,
            stream_a_enabled: s.stream_a_enabled,
            stream_b_enabled: s.stream_b_enabled,
            error: s.error,
        }
    }
}

/// Initialize the audio engine with the DeepFilter model
/// Creates shared ONNX sessions and two independent stream processors
#[frb]
pub fn init_engine(model_data: Vec<u8>) -> Result<(), String> {
    deepfilter_audio::api::init_engine(model_data)
}

/// Enable or disable stream A
#[frb]
pub fn set_stream_a_enabled(enabled: bool) -> Result<(), String> {
    deepfilter_audio::api::set_stream_a_enabled(enabled)
}

/// Enable or disable stream B
#[frb]
pub fn set_stream_b_enabled(enabled: bool) -> Result<(), String> {
    deepfilter_audio::api::set_stream_b_enabled(enabled)
}

/// Check if stream A is enabled
#[frb]
pub fn is_stream_a_enabled() -> bool {
    deepfilter_audio::api::is_stream_a_enabled()
}

/// Check if stream B is enabled
#[frb]
pub fn is_stream_b_enabled() -> bool {
    deepfilter_audio::api::is_stream_b_enabled()
}

/// Start recording audio to the specified file path
#[frb]
pub fn start_recording(output_path: String) -> Result<(), String> {
    deepfilter_audio::api::start_recording(output_path)
}

/// Stop recording and save the file
#[frb]
pub fn stop_recording() -> Result<String, String> {
    deepfilter_audio::api::stop_recording()
}

/// Play audio from the specified file path
#[frb]
pub fn start_playback(file_path: String) -> Result<(), String> {
    deepfilter_audio::api::start_playback(file_path)
}

/// Stop playback
#[frb]
pub fn stop_playback() -> Result<(), String> {
    deepfilter_audio::api::stop_playback()
}

/// Get current recording/playback status
#[frb]
pub fn get_status() -> RecordingStatus {
    deepfilter_audio::api::get_status().into()
}

/// Check if playback has finished
#[frb]
pub fn is_playback_finished() -> bool {
    deepfilter_audio::api::is_playback_finished()
}

/// System metrics for UI display
pub struct SystemMetrics {
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub nnapi_available: bool,
}

impl From<deepfilter_audio::api::SystemMetrics> for SystemMetrics {
    fn from(m: deepfilter_audio::api::SystemMetrics) -> Self {
        Self {
            cpu_usage_percent: m.cpu_usage_percent,
            gpu_usage_percent: m.gpu_usage_percent,
            nnapi_available: m.nnapi_available,
        }
    }
}

/// Get current system metrics (CPU/GPU usage)
#[frb]
pub fn get_system_metrics() -> SystemMetrics {
    deepfilter_audio::api::get_system_metrics().into()
}
