use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use anyhow::{Result, Context};

use super::{SAMPLE_RATE, FRAME_SIZE};
use crate::io::WavReader;

pub struct AudioPlayer {
    is_playing: Arc<AtomicBool>,
    is_finished: Arc<AtomicBool>,
    samples: Arc<Vec<f32>>,
    position: Arc<AtomicUsize>,
    #[cfg(not(target_os = "android"))]
    stream: Option<cpal::Stream>,
    // On Android, stream is managed via Oboe callbacks
}

// AudioPlayer is Send-safe on Android (no non-Send fields)
#[cfg(target_os = "android")]
unsafe impl Send for AudioPlayer {}

impl AudioPlayer {
    pub fn new(file_path: &str) -> Result<Self> {
        let reader = WavReader::new(file_path)?;
        let samples = reader.read_all_samples()?;

        log::info!("Loaded {} samples from WAV file", samples.len());

        Ok(Self {
            is_playing: Arc::new(AtomicBool::new(false)),
            is_finished: Arc::new(AtomicBool::new(false)),
            samples: Arc::new(samples),
            position: Arc::new(AtomicUsize::new(0)),
            #[cfg(not(target_os = "android"))]
            stream: None,
        })
    }

    #[cfg(target_os = "android")]
    pub fn start(&mut self) -> Result<()> {
        use oboe::{
            AudioOutputCallback, AudioOutputStreamSafe, AudioStream, AudioStreamBuilder,
            DataCallbackResult, Mono, Output, PerformanceMode, SharingMode,
        };

        struct PlayerCallback {
            is_playing: Arc<AtomicBool>,
            is_finished: Arc<AtomicBool>,
            samples: Arc<Vec<f32>>,
            position: Arc<AtomicUsize>,
        }

        impl AudioOutputCallback for PlayerCallback {
            type FrameType = (i16, Mono);

            fn on_audio_ready(
                &mut self,
                _stream: &mut dyn AudioOutputStreamSafe,
                audio_data: &mut [i16],
            ) -> DataCallbackResult {
                if !self.is_playing.load(Ordering::SeqCst) {
                    return DataCallbackResult::Stop;
                }

                let pos = self.position.load(Ordering::SeqCst);
                let samples_len = self.samples.len();

                if pos >= samples_len {
                    audio_data.fill(0);
                    self.is_finished.store(true, Ordering::SeqCst);
                    self.is_playing.store(false, Ordering::SeqCst);
                    return DataCallbackResult::Stop;
                }

                let samples_to_copy = audio_data.len().min(samples_len - pos);
                // Convert f32 to i16 for output
                for (i, sample) in self.samples[pos..pos + samples_to_copy].iter().enumerate() {
                    audio_data[i] = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                }

                if samples_to_copy < audio_data.len() {
                    audio_data[samples_to_copy..].fill(0);
                }

                self.position.fetch_add(samples_to_copy, Ordering::SeqCst);
                DataCallbackResult::Continue
            }
        }

        let callback = PlayerCallback {
            is_playing: self.is_playing.clone(),
            is_finished: self.is_finished.clone(),
            samples: self.samples.clone(),
            position: self.position.clone(),
        };

        self.is_playing.store(true, Ordering::SeqCst);
        self.is_finished.store(false, Ordering::SeqCst);
        self.position.store(0, Ordering::SeqCst);

        let mut stream = AudioStreamBuilder::default()
            .set_direction::<Output>()
            .set_performance_mode(PerformanceMode::None)  // Use legacy mode for better Samsung compatibility
            .set_sharing_mode(SharingMode::Shared)  // Shared mode for better compatibility
            .set_sample_rate(SAMPLE_RATE as i32)
            .set_channel_count::<Mono>()
            .set_format::<i16>()  // Use i16 for better device compatibility
            .set_frames_per_callback(FRAME_SIZE as i32)
            .set_callback(callback)
            .open_stream()
            .context("Failed to open audio output stream")?;

        stream.start().context("Failed to start playback stream")?;

        // Stream is managed by Oboe callback
        std::mem::forget(stream);

        log::info!("Audio playback started (Android/Oboe)");
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    pub fn start(&mut self) -> Result<()> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host.default_output_device()
            .context("No output device available")?;

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Fixed(FRAME_SIZE as u32),
        };

        let is_playing = self.is_playing.clone();
        let is_finished = self.is_finished.clone();
        let samples = self.samples.clone();
        let position = self.position.clone();

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if !is_playing.load(Ordering::SeqCst) {
                    data.fill(0.0);
                    return;
                }

                let pos = position.load(Ordering::SeqCst);
                let samples_len = samples.len();

                if pos >= samples_len {
                    data.fill(0.0);
                    is_finished.store(true, Ordering::SeqCst);
                    is_playing.store(false, Ordering::SeqCst);
                    return;
                }

                let samples_to_copy = data.len().min(samples_len - pos);
                data[..samples_to_copy].copy_from_slice(&samples[pos..pos + samples_to_copy]);

                if samples_to_copy < data.len() {
                    data[samples_to_copy..].fill(0.0);
                }

                position.fetch_add(samples_to_copy, Ordering::SeqCst);
            },
            |err| log::error!("Audio output error: {}", err),
            None,
        ).context("Failed to build output stream")?;

        self.is_playing.store(true, Ordering::SeqCst);
        self.is_finished.store(false, Ordering::SeqCst);
        self.position.store(0, Ordering::SeqCst);

        stream.play().context("Failed to start playback stream")?;
        self.stream = Some(stream);

        log::info!("Audio playback started (Desktop/CPAL)");
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.is_playing.store(false, Ordering::SeqCst);

        // On Android, setting is_playing to false causes callback to stop
        #[cfg(not(target_os = "android"))]
        {
            self.stream = None;
        }

        log::info!("Audio playback stopped");
        Ok(())
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::SeqCst)
    }

    pub fn is_finished(&self) -> bool {
        self.is_finished.load(Ordering::SeqCst)
    }
}
