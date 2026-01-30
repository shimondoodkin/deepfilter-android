use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use flutter_rust_bridge::frb;

use crate::audio::{AudioRecorder, AudioPlayer, SAMPLE_RATE, FRAME_SIZE};
use crate::processing::{SharedSessions, StreamProcessor};
use crate::io::WavWriter;

/// Recording status for UI updates
#[frb(dart_metadata=("freezed"))]
pub struct RecordingStatus {
    pub is_recording: bool,
    pub is_playing: bool,
    pub duration_ms: u64,
    pub stream_a_enabled: bool,
    pub stream_b_enabled: bool,
    pub error: Option<String>,
}

/// Store model bytes globally (thread-safe)
static MODEL_DATA: Mutex<Option<Vec<u8>>> = Mutex::new(None);
static MODEL_LOADED: AtomicBool = AtomicBool::new(false);

/// Shared ONNX sessions (loaded once, shared by all streams)
static SHARED_SESSIONS: Mutex<Option<Arc<SharedSessions>>> = Mutex::new(None);

/// Dual stream processors
static STREAM_A: Mutex<Option<StreamProcessor>> = Mutex::new(None);
static STREAM_B: Mutex<Option<StreamProcessor>> = Mutex::new(None);

/// Store recorder state
static RECORDER_PATH: Mutex<Option<String>> = Mutex::new(None);
static IS_RECORDING: AtomicBool = AtomicBool::new(false);
static RECORDING_DURATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Store player state
static IS_PLAYING: AtomicBool = AtomicBool::new(false);
static PLAYER_FINISHED: AtomicBool = AtomicBool::new(true);

// Global storage for recorder/player
static RECORDER: Mutex<Option<AudioRecorder>> = Mutex::new(None);
static PLAYER: Mutex<Option<AudioPlayer>> = Mutex::new(None);

/// Initialize the audio engine with the DeepFilter model
/// Creates shared ONNX sessions and two independent stream processors
#[frb]
pub fn init_engine(model_data: Vec<u8>) -> Result<(), String> {
    log::info!("Initializing engine with DeepFilter model ({} bytes)", model_data.len());

    // Load shared ONNX sessions
    let sessions = SharedSessions::new(&model_data)
        .map_err(|e| format!("Failed to load DeepFilter model: {}", e))?;

    // Create two stream processors sharing the same sessions
    let stream_a = StreamProcessor::new(Arc::clone(&sessions), 0)
        .map_err(|e| format!("Failed to create stream A: {}", e))?;
    let stream_b = StreamProcessor::new(Arc::clone(&sessions), 1)
        .map_err(|e| format!("Failed to create stream B: {}", e))?;

    // Store everything
    *MODEL_DATA.lock() = Some(model_data);
    *SHARED_SESSIONS.lock() = Some(sessions);
    *STREAM_A.lock() = Some(stream_a);
    *STREAM_B.lock() = Some(stream_b);
    MODEL_LOADED.store(true, Ordering::SeqCst);

    log::info!("Engine initialized with dual streams (A and B)");
    Ok(())
}

/// Enable or disable stream A
#[frb]
pub fn set_stream_a_enabled(enabled: bool) -> Result<(), String> {
    let guard = STREAM_A.lock();
    if let Some(ref stream) = *guard {
        stream.set_enabled(enabled);
        Ok(())
    } else {
        Err("Stream A not initialized".to_string())
    }
}

/// Enable or disable stream B
#[frb]
pub fn set_stream_b_enabled(enabled: bool) -> Result<(), String> {
    let guard = STREAM_B.lock();
    if let Some(ref stream) = *guard {
        stream.set_enabled(enabled);
        Ok(())
    } else {
        Err("Stream B not initialized".to_string())
    }
}

/// Check if stream A is enabled
#[frb]
pub fn is_stream_a_enabled() -> bool {
    STREAM_A.lock().as_ref().map(|s| s.is_enabled()).unwrap_or(false)
}

/// Check if stream B is enabled
#[frb]
pub fn is_stream_b_enabled() -> bool {
    STREAM_B.lock().as_ref().map(|s| s.is_enabled()).unwrap_or(false)
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

    // Reset stream states for new recording
    if let Some(ref mut stream) = *STREAM_A.lock() {
        stream.reset();
    }
    if let Some(ref mut stream) = *STREAM_B.lock() {
        stream.reset();
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
/// Processes audio through both streams (if enabled) and saves combined output
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

    log::info!("Processing {} samples through dual streams...", raw_samples.len());

    // Process through both streams
    let mut output_a = Vec::with_capacity(raw_samples.len());
    let mut output_b = Vec::with_capacity(raw_samples.len());

    let mut input_buffer = Vec::with_capacity(FRAME_SIZE);
    let mut output_buffer_a = vec![0.0f32; FRAME_SIZE];
    let mut output_buffer_b = vec![0.0f32; FRAME_SIZE];

    // Lock both streams for processing
    let mut stream_a_guard = STREAM_A.lock();
    let mut stream_b_guard = STREAM_B.lock();

    for sample in raw_samples.iter() {
        input_buffer.push(*sample);

        if input_buffer.len() >= FRAME_SIZE {
            // Process through stream A
            if let Some(stream) = stream_a_guard.as_mut() {
                match stream.process_frame(&input_buffer, &mut output_buffer_a) {
                    Ok(_) => output_a.extend_from_slice(&output_buffer_a),
                    Err(e) => {
                        log::error!("Stream A processing error: {}", e);
                        output_a.extend_from_slice(&input_buffer);
                    }
                }
            } else {
                output_a.extend_from_slice(&input_buffer);
            }

            // Process through stream B (same input)
            if let Some(stream) = stream_b_guard.as_mut() {
                match stream.process_frame(&input_buffer, &mut output_buffer_b) {
                    Ok(_) => output_b.extend_from_slice(&output_buffer_b),
                    Err(e) => {
                        log::error!("Stream B processing error: {}", e);
                        output_b.extend_from_slice(&input_buffer);
                    }
                }
            } else {
                output_b.extend_from_slice(&input_buffer);
            }

            input_buffer.clear();
        }
    }

    // Handle remaining samples
    if !input_buffer.is_empty() {
        input_buffer.resize(FRAME_SIZE, 0.0);

        if let Some(stream) = stream_a_guard.as_mut() {
            match stream.process_frame(&input_buffer, &mut output_buffer_a) {
                Ok(_) => output_a.extend_from_slice(&output_buffer_a),
                Err(_) => output_a.extend_from_slice(&input_buffer),
            }
        }

        if let Some(stream) = stream_b_guard.as_mut() {
            match stream.process_frame(&input_buffer, &mut output_buffer_b) {
                Ok(_) => output_b.extend_from_slice(&output_buffer_b),
                Err(_) => output_b.extend_from_slice(&input_buffer),
            }
        }
    }

    // Release locks
    drop(stream_a_guard);
    drop(stream_b_guard);

    // Mix the two streams (average)
    let processed_samples: Vec<f32> = output_a.iter()
        .zip(output_b.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect();

    log::info!(
        "Processed: stream_a={} samples, stream_b={} samples, mixed={} samples",
        output_a.len(), output_b.len(), processed_samples.len()
    );

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

    let stream_a_enabled = STREAM_A.lock().as_ref().map(|s| s.is_enabled()).unwrap_or(false);
    let stream_b_enabled = STREAM_B.lock().as_ref().map(|s| s.is_enabled()).unwrap_or(false);

    RecordingStatus {
        is_recording,
        is_playing,
        duration_ms,
        stream_a_enabled,
        stream_b_enabled,
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
