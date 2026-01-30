// Wrapper API for flutter_rust_bridge
// Re-exports deepfilter_audio functions with #[frb] annotations

use flutter_rust_bridge::frb;

/// Recording status for UI updates
pub struct RecordingStatus {
    pub is_recording: bool,
    pub is_playing: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
}

impl From<deepfilter_audio::api::RecordingStatus> for RecordingStatus {
    fn from(s: deepfilter_audio::api::RecordingStatus) -> Self {
        Self {
            is_recording: s.is_recording,
            is_playing: s.is_playing,
            duration_ms: s.duration_ms,
            error: s.error,
        }
    }
}

/// Initialize the audio engine with the DeepFilter model
#[frb]
pub fn init_engine(model_data: Vec<u8>) -> Result<(), String> {
    deepfilter_audio::api::init_engine(model_data)
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
