use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::Mutex;
use flutter_rust_bridge::frb;

use crate::audio::{AudioRecorder, AudioPlayer, SAMPLE_RATE, FRAME_SIZE};
use crate::processing::DeepFilter;
use crate::io::WavWriter;

/// Recording status for UI updates
#[frb(dart_metadata=("freezed"))]
pub struct RecordingStatus {
    pub is_recording: bool,
    pub is_playing: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
}

/// Store model bytes globally (thread-safe)
static MODEL_DATA: Mutex<Option<Vec<u8>>> = Mutex::new(None);
static MODEL_LOADED: AtomicBool = AtomicBool::new(false);

/// Store recorder state
static RECORDER_PATH: Mutex<Option<String>> = Mutex::new(None);
static IS_RECORDING: AtomicBool = AtomicBool::new(false);
static RECORDING_DURATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Store player state
static IS_PLAYING: AtomicBool = AtomicBool::new(false);
static PLAYER_FINISHED: AtomicBool = AtomicBool::new(true);

// Global storage for recorder (now Send-safe on Android)
static RECORDER: Mutex<Option<AudioRecorder>> = Mutex::new(None);
static PLAYER: Mutex<Option<AudioPlayer>> = Mutex::new(None);

/// Initialize the audio engine with the DeepFilter model
#[frb]
pub fn init_engine(model_data: Vec<u8>) -> Result<(), String> {
    log::info!("Storing DeepFilter model ({} bytes)", model_data.len());

    // Validate model can be loaded
    match DeepFilter::new(&model_data) {
        Ok(_) => {
            *MODEL_DATA.lock() = Some(model_data);
            MODEL_LOADED.store(true, Ordering::SeqCst);
            log::info!("DeepFilter model validated and stored");
            Ok(())
        }
        Err(e) => {
            let msg = format!("Failed to load DeepFilter model: {}", e);
            log::error!("{}", msg);
            Err(msg)
        }
    }
}

/// Start recording audio to the specified file path
#[frb]
pub fn start_recording(output_path: String) -> Result<(), String> {
    log::info!("Starting recording to: {}", output_path);

    if IS_RECORDING.load(Ordering::SeqCst) {
        return Err("Already recording".to_string());
    }

    if !MODEL_LOADED.load(Ordering::SeqCst) {
        return Err("DeepFilter model not loaded. Call init_engine first.".to_string());
    }

    let mut recorder = AudioRecorder::new(output_path.clone())
        .map_err(|e| format!("Failed to create recorder: {}", e))?;

    recorder.start().map_err(|e| e.to_string())?;

    *RECORDER_PATH.lock() = Some(output_path);
    IS_RECORDING.store(true, Ordering::SeqCst);
    RECORDING_DURATION.store(0, Ordering::SeqCst);

    *RECORDER.lock() = Some(recorder);

    log::info!("Recording started");
    Ok(())
}

/// Stop recording and save the file
#[frb]
pub fn stop_recording() -> Result<String, String> {
    log::info!("Stopping recording");

    if !IS_RECORDING.load(Ordering::SeqCst) {
        return Err("Not recording".to_string());
    }

    // Get the recorder and stop it
    let (raw_samples, output_path) = {
        let mut guard = RECORDER.lock();
        if let Some(mut recorder) = guard.take() {
            recorder.stop().map_err(|e| e.to_string())?
        } else {
            return Err("Recorder not found".to_string());
        }
    };

    IS_RECORDING.store(false, Ordering::SeqCst);

    // Now process with DeepFilter (on this thread)
    log::info!("Processing {} samples through DeepFilter...", raw_samples.len());

    let model_data = MODEL_DATA.lock().clone()
        .ok_or("Model data not available")?;

    let mut deep_filter = DeepFilter::new(&model_data)
        .map_err(|e| format!("Failed to create DeepFilter: {}", e))?;

    let mut processed_samples = Vec::with_capacity(raw_samples.len());
    let mut input_buffer = Vec::with_capacity(FRAME_SIZE);
    let mut output_buffer = vec![0.0f32; FRAME_SIZE];

    for sample in raw_samples.iter() {
        input_buffer.push(*sample);

        if input_buffer.len() >= FRAME_SIZE {
            match deep_filter.process_frame(&input_buffer, &mut output_buffer) {
                Ok(_lsnr) => {
                    processed_samples.extend_from_slice(&output_buffer);
                }
                Err(e) => {
                    log::error!("DeepFilter processing error: {}", e);
                    // On error, use unprocessed audio
                    processed_samples.extend_from_slice(&input_buffer);
                }
            }
            input_buffer.clear();
        }
    }

    // Handle remaining samples (pad with zeros if needed)
    if !input_buffer.is_empty() {
        input_buffer.resize(FRAME_SIZE, 0.0);
        match deep_filter.process_frame(&input_buffer, &mut output_buffer) {
            Ok(_) => processed_samples.extend_from_slice(&output_buffer),
            Err(_) => processed_samples.extend_from_slice(&input_buffer),
        }
    }

    // Write processed audio to WAV file
    let mut writer = WavWriter::new(&output_path, SAMPLE_RATE, 1)
        .map_err(|e| format!("Failed to create WAV writer: {}", e))?;
    writer.write_samples(&processed_samples)
        .map_err(|e| format!("Failed to write samples: {}", e))?;
    writer.finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    log::info!("Recording saved to: {} ({} samples)", output_path, processed_samples.len());
    Ok(output_path)
}

/// Play audio from the specified file path
#[frb]
pub fn start_playback(file_path: String) -> Result<(), String> {
    log::info!("Starting playback of: {}", file_path);

    if IS_PLAYING.load(Ordering::SeqCst) {
        return Err("Already playing".to_string());
    }

    let mut player = AudioPlayer::new(&file_path)
        .map_err(|e| format!("Failed to create player: {}", e))?;

    player.start().map_err(|e| e.to_string())?;

    IS_PLAYING.store(true, Ordering::SeqCst);
    PLAYER_FINISHED.store(false, Ordering::SeqCst);

    *PLAYER.lock() = Some(player);

    log::info!("Playback started");
    Ok(())
}

/// Stop playback
#[frb]
pub fn stop_playback() -> Result<(), String> {
    log::info!("Stopping playback");

    if !IS_PLAYING.load(Ordering::SeqCst) {
        return Err("Not playing".to_string());
    }

    let result = {
        let mut guard = PLAYER.lock();
        if let Some(mut player) = guard.take() {
            player.stop()
        } else {
            Err(anyhow::anyhow!("Player not found"))
        }
    };

    IS_PLAYING.store(false, Ordering::SeqCst);

    match result {
        Ok(()) => {
            log::info!("Playback stopped");
            Ok(())
        }
        Err(e) => {
            let msg = format!("Failed to stop playback: {}", e);
            log::error!("{}", msg);
            Err(msg)
        }
    }
}

/// Get current recording/playback status
#[frb]
pub fn get_status() -> RecordingStatus {
    let is_recording = IS_RECORDING.load(Ordering::SeqCst);
    let is_playing = IS_PLAYING.load(Ordering::SeqCst);

    let duration_ms = RECORDER.lock().as_ref().map(|rec| rec.duration_ms()).unwrap_or(0);

    RecordingStatus {
        is_recording,
        is_playing,
        duration_ms,
        error: None,
    }
}

/// Check if playback has finished
#[frb]
pub fn is_playback_finished() -> bool {
    let finished = PLAYER.lock().as_ref().map(|player| player.is_finished()).unwrap_or(true);

    if finished {
        IS_PLAYING.store(false, Ordering::SeqCst);
        PLAYER_FINISHED.store(true, Ordering::SeqCst);
    }

    finished
}
