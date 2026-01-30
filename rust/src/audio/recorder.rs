use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;
use anyhow::{Result, Context};

use super::{SAMPLE_RATE, FRAME_SIZE};

/// Audio recorder that captures raw samples (without DeepFilter processing)
/// DeepFilter processing is done separately when stop is called
pub struct AudioRecorder {
    output_path: String,
    is_recording: Arc<AtomicBool>,
    duration_samples: Arc<AtomicU64>,
    raw_samples: Arc<Mutex<Vec<f32>>>,
    #[cfg(not(target_os = "android"))]
    stream: Option<cpal::Stream>,
    // On Android, the stream is managed internally by Oboe
    // We just track state through atomic flags
}

// AudioRecorder is Send-safe because:
// - On Android: no non-Send fields
// - On desktop: cpal::Stream is not Send, but we only access it while holding the Mutex lock
unsafe impl Send for AudioRecorder {}
// AudioRecorder is Sync-safe because all access goes through a Mutex
unsafe impl Sync for AudioRecorder {}

impl AudioRecorder {
    pub fn new(output_path: String) -> Result<Self> {
        Ok(Self {
            output_path,
            is_recording: Arc::new(AtomicBool::new(false)),
            duration_samples: Arc::new(AtomicU64::new(0)),
            raw_samples: Arc::new(Mutex::new(Vec::new())),
            #[cfg(not(target_os = "android"))]
            stream: None,
        })
    }

    #[cfg(not(target_os = "android"))]
    pub fn start(&mut self) -> Result<()> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        // Clear raw samples buffer
        self.raw_samples.lock().clear();

        let host = cpal::default_host();
        let device = host.default_input_device()
            .context("No input device available")?;

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Default,
        };

        let is_recording = self.is_recording.clone();
        let duration_samples = self.duration_samples.clone();
        let raw_samples = self.raw_samples.clone();

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !is_recording.load(Ordering::SeqCst) {
                    return;
                }

                // Store raw samples for later processing
                raw_samples.lock().extend_from_slice(data);
                duration_samples.fetch_add(data.len() as u64, Ordering::SeqCst);
            },
            |err| log::error!("Audio input error: {}", err),
            None,
        ).context("Failed to build input stream")?;

        self.is_recording.store(true, Ordering::SeqCst);
        self.duration_samples.store(0, Ordering::SeqCst);

        stream.play().context("Failed to start audio stream")?;
        self.stream = Some(stream);

        log::info!("Audio recording started (Desktop/CPAL)");
        Ok(())
    }

    #[cfg(target_os = "android")]
    pub fn start(&mut self) -> Result<()> {
        use oboe::{
            AudioInputCallback, AudioInputStreamSafe, AudioStream, AudioStreamBuilder,
            DataCallbackResult, Input, Mono, PerformanceMode, SharingMode,
        };

        // Clear raw samples buffer
        self.raw_samples.lock().clear();

        struct RecorderCallback {
            is_recording: Arc<AtomicBool>,
            duration_samples: Arc<AtomicU64>,
            raw_samples: Arc<Mutex<Vec<f32>>>,
        }

        impl AudioInputCallback for RecorderCallback {
            type FrameType = (i16, Mono);

            fn on_audio_ready(
                &mut self,
                _stream: &mut dyn AudioInputStreamSafe,
                audio_data: &[i16],
            ) -> DataCallbackResult {
                if !self.is_recording.load(Ordering::SeqCst) {
                    return DataCallbackResult::Stop;
                }

                // Convert i16 to f32 for processing
                let float_samples: Vec<f32> = audio_data
                    .iter()
                    .map(|&s| s as f32 / 32768.0)
                    .collect();
                self.raw_samples.lock().extend_from_slice(&float_samples);
                self.duration_samples.fetch_add(audio_data.len() as u64, Ordering::SeqCst);

                DataCallbackResult::Continue
            }
        }

        let callback = RecorderCallback {
            is_recording: self.is_recording.clone(),
            duration_samples: self.duration_samples.clone(),
            raw_samples: self.raw_samples.clone(),
        };

        self.is_recording.store(true, Ordering::SeqCst);
        self.duration_samples.store(0, Ordering::SeqCst);

        let mut stream = AudioStreamBuilder::default()
            .set_direction::<Input>()
            .set_performance_mode(PerformanceMode::None)  // Use legacy mode for better Samsung compatibility
            .set_sharing_mode(SharingMode::Shared)  // Shared mode for better compatibility
            .set_sample_rate(SAMPLE_RATE as i32)
            .set_channel_count::<Mono>()
            .set_format::<i16>()  // Use i16 for better device compatibility
            .set_frames_per_callback(FRAME_SIZE as i32)
            .set_callback(callback)
            .open_stream()
            .context("Failed to open audio input stream")?;

        stream.start().context("Failed to start audio stream")?;

        // Stream is managed by Oboe callback, it will stop when callback returns Stop
        // We leak the stream intentionally - it cleans up when callback stops
        std::mem::forget(stream);

        log::info!("Audio recording started (Android/Oboe)");
        Ok(())
    }

    /// Stop recording and return the raw samples and output path
    /// The caller is responsible for processing with DeepFilter and saving
    pub fn stop(&mut self) -> Result<(Vec<f32>, String)> {
        self.is_recording.store(false, Ordering::SeqCst);

        // On Android, setting is_recording to false causes the callback to return Stop
        // which stops the stream. On desktop, we drop the stream.
        #[cfg(not(target_os = "android"))]
        {
            self.stream = None;
        }

        // Give the stream a moment to stop
        std::thread::sleep(std::time::Duration::from_millis(100));

        let raw_samples = self.raw_samples.lock().clone();
        log::info!("Recording stopped, captured {} samples", raw_samples.len());

        Ok((raw_samples, self.output_path.clone()))
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }

    pub fn duration_ms(&self) -> u64 {
        let samples = self.duration_samples.load(Ordering::SeqCst);
        (samples * 1000) / SAMPLE_RATE as u64
    }
}
