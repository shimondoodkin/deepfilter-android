use anyhow::{Result, Context};
use ndarray::{Array2, ArrayView2};
use df::tract::{DfParams, DfTract, RuntimeParams};

use crate::audio::FRAME_SIZE;

/// DeepFilterNet wrapper for noise suppression
pub struct DeepFilter {
    df: DfTract,
}

impl DeepFilter {
    /// Create a new DeepFilter instance from model bytes
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        log::info!("Loading DeepFilter model ({} bytes)", model_bytes.len());

        // Leak the model bytes to get a 'static lifetime
        // This is acceptable for a test app - the model lives for the entire app lifetime
        let model_bytes_static: &'static [u8] = Box::leak(model_bytes.to_vec().into_boxed_slice());

        let params = DfParams::from_bytes(model_bytes_static)
            .context("Failed to parse DeepFilter model")?;

        let runtime = RuntimeParams::default_with_ch(1)
            .with_atten_lim(100.0);  // No attenuation limit

        let df = DfTract::new(params, &runtime)
            .context("Failed to initialize DeepFilter")?;

        log::info!(
            "DeepFilter initialized: sr={}, hop_size={}, fft_size={}",
            df.sr, df.hop_size, df.fft_size
        );

        Ok(Self { df })
    }

    /// Process a single frame of audio
    /// Input and output must be FRAME_SIZE samples
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32> {
        if input.len() != FRAME_SIZE || output.len() != FRAME_SIZE {
            anyhow::bail!(
                "Invalid frame size: input={}, output={}, expected={}",
                input.len(), output.len(), FRAME_SIZE
            );
        }

        // Create ndarray views for DeepFilter
        let input_arr = ArrayView2::from_shape((1, FRAME_SIZE), input)
            .context("Failed to create input array view")?;

        let mut output_arr = Array2::<f32>::zeros((1, FRAME_SIZE));

        // Process through DeepFilter
        let lsnr = self.df.process(input_arr, output_arr.view_mut())
            .context("DeepFilter processing failed")?;

        // Copy to output slice
        output.copy_from_slice(output_arr.as_slice().unwrap());

        Ok(lsnr)
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.df.sr
    }

    /// Get the hop size (frame size)
    pub fn hop_size(&self) -> usize {
        self.df.hop_size
    }
}
