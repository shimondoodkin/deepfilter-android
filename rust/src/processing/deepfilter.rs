use anyhow::{Context, Result, bail};
use std::io::{Cursor, Read};
use std::sync::{Arc, Once};
use std::sync::atomic::{AtomicBool, Ordering};
use flate2::read::GzDecoder;
use tar::Archive;
use ini::Ini;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Tensor, Value};
use num_complex::Complex32;

use df::DFState;

// Initialize ORT once
static ORT_INIT: Once = Once::new();

/// Initialize ONNX Runtime (must be called before creating sessions)
fn init_ort() -> Result<()> {
    let mut init_result: Result<()> = Ok(());

    ORT_INIT.call_once(|| {
        log::info!("Initializing ONNX Runtime...");

        #[cfg(target_os = "android")]
        {
            match ort::init_from("libonnxruntime.so") {
                Ok(_) => log::info!("ONNX Runtime initialized successfully"),
                Err(e) => {
                    log::error!("Failed to initialize ONNX Runtime: {}", e);
                    init_result = Err(anyhow::anyhow!("ORT init failed: {}", e));
                }
            }
        }

        #[cfg(not(target_os = "android"))]
        {
            let _ = ort::init();
            log::info!("ONNX Runtime initialized (desktop)");
        }
    });

    init_result
}

/// DeepFilter model configuration
#[derive(Clone)]
pub struct DfConfig {
    pub sr: usize,
    pub hop_size: usize,
    pub fft_size: usize,
    pub nb_erb: usize,
    pub nb_df: usize,
    pub min_nb_erb_freqs: usize,
    pub df_order: usize,
    pub conv_lookahead: usize,
    pub df_lookahead: usize,
    pub alpha: f32,
}

/// Shared ONNX Runtime sessions - thread-safe, can be shared across streams
/// Sessions wrapped in Mutex because Session::run requires &mut self
pub struct SharedSessions {
    pub enc_session: parking_lot::Mutex<Session>,
    pub erb_dec_session: parking_lot::Mutex<Session>,
    pub df_dec_session: parking_lot::Mutex<Session>,
    pub config: DfConfig,
}

impl SharedSessions {
    /// Load shared sessions from model bytes (tar.gz archive)
    pub fn new(model_bytes: &[u8]) -> Result<Arc<Self>> {
        log::info!("Loading DeepFilter model with ONNX Runtime ({} bytes)", model_bytes.len());

        // Initialize ONNX Runtime
        init_ort()?;

        // Parse tar.gz archive
        let tar = GzDecoder::new(Cursor::new(model_bytes));
        let mut archive = Archive::new(tar);

        let mut enc_bytes = Vec::new();
        let mut erb_dec_bytes = Vec::new();
        let mut df_dec_bytes = Vec::new();
        let mut config_str = String::new();

        for entry in archive.entries().context("Failed to read model archive")? {
            let mut file = entry.context("Failed to read archive entry")?;
            let path = file.path()?.to_path_buf();

            let filename = path.file_name()
                .map(|f| f.to_string_lossy())
                .unwrap_or_default();

            if filename.starts_with("._") {
                continue;
            }

            if filename == "enc.onnx" {
                file.read_to_end(&mut enc_bytes)?;
                log::info!("Loaded encoder: {} bytes", enc_bytes.len());
            } else if filename == "erb_dec.onnx" {
                file.read_to_end(&mut erb_dec_bytes)?;
                log::info!("Loaded ERB decoder: {} bytes", erb_dec_bytes.len());
            } else if filename == "df_dec.onnx" {
                file.read_to_end(&mut df_dec_bytes)?;
                log::info!("Loaded DF decoder: {} bytes", df_dec_bytes.len());
            } else if filename == "config.ini" {
                let mut config_bytes = Vec::new();
                file.read_to_end(&mut config_bytes)?;
                config_str = String::from_utf8_lossy(&config_bytes).into_owned();
                log::info!("Loaded config.ini: {} bytes", config_str.len());
            }
        }

        if config_str.is_empty() {
            bail!("config.ini not found in model archive");
        }

        let config_ini = Ini::load_from_str(&config_str)
            .context("Failed to parse config.ini")?;

        if enc_bytes.is_empty() || erb_dec_bytes.is_empty() || df_dec_bytes.is_empty() {
            bail!("Model archive missing required ONNX files");
        }

        // Parse configuration
        let df_cfg = config_ini.section(Some("df"))
            .context("Missing [df] section in config")?;
        let model_cfg = config_ini.section(Some("deepfilternet"))
            .context("Missing [deepfilternet] section in config")?;

        let sr: usize = df_cfg.get("sr").context("Missing sr")?.parse()?;
        let hop_size: usize = df_cfg.get("hop_size").context("Missing hop_size")?.parse()?;
        let fft_size: usize = df_cfg.get("fft_size").context("Missing fft_size")?.parse()?;
        let nb_erb: usize = df_cfg.get("nb_erb").context("Missing nb_erb")?.parse()?;
        let nb_df: usize = df_cfg.get("nb_df").context("Missing nb_df")?.parse()?;
        let min_nb_erb_freqs: usize = df_cfg.get("min_nb_erb_freqs").context("Missing min_nb_erb_freqs")?.parse()?;
        let df_order: usize = df_cfg.get("df_order")
            .or_else(|| model_cfg.get("df_order"))
            .context("Missing df_order")?.parse()?;
        let conv_lookahead: usize = model_cfg.get("conv_lookahead").context("Missing conv_lookahead")?.parse()?;
        let df_lookahead: usize = df_cfg.get("df_lookahead")
            .or_else(|| model_cfg.get("df_lookahead"))
            .context("Missing df_lookahead")?.parse()?;

        let alpha = if let Some(a) = df_cfg.get("norm_alpha") {
            a.parse::<f32>()?
        } else {
            let tau: f32 = df_cfg.get("norm_tau").context("Missing norm_tau")?.parse()?;
            calc_norm_alpha(sr, hop_size, tau)
        };

        let config = DfConfig {
            sr,
            hop_size,
            fft_size,
            nb_erb,
            nb_df,
            min_nb_erb_freqs,
            df_order,
            conv_lookahead,
            df_lookahead,
            alpha,
        };

        log::info!(
            "DeepFilter config: sr={}, hop_size={}, fft_size={}, nb_erb={}, nb_df={}, df_order={}",
            sr, hop_size, fft_size, nb_erb, nb_df, df_order
        );

        // Create ONNX Runtime sessions
        let enc_session = Self::create_session(&enc_bytes, "encoder")?;
        let erb_dec_session = Self::create_session(&erb_dec_bytes, "erb_decoder")?;
        let df_dec_session = Self::create_session(&df_dec_bytes, "df_decoder")?;

        log::info!("SharedSessions initialized with ONNX Runtime");

        Ok(Arc::new(Self {
            enc_session: parking_lot::Mutex::new(enc_session),
            erb_dec_session: parking_lot::Mutex::new(erb_dec_session),
            df_dec_session: parking_lot::Mutex::new(df_dec_session),
            config,
        }))
    }

    fn create_session(model_bytes: &[u8], name: &str) -> Result<Session> {
        log::info!("Creating ONNX session for {} ({} bytes)", name, model_bytes.len());

        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Session builder failed for {}: {}", name, e))?;

        // Try to use NNAPI on Android for GPU/NPU acceleration
        #[cfg(target_os = "android")]
        let builder = {
            use ort::execution_providers::NNAPIExecutionProvider;
            use crate::api::metrics::set_nnapi_registered;

            let nnapi = NNAPIExecutionProvider::default();
            match builder.with_execution_providers([nnapi.build()]) {
                Ok(b) => {
                    log::info!("NNAPI execution provider registered for {}", name);
                    set_nnapi_registered(true);
                    b
                }
                Err(e) => {
                    log::warn!("Failed to register NNAPI for {}: {} - falling back to CPU", name, e);
                    set_nnapi_registered(false);
                    Session::builder()
                        .map_err(|e| anyhow::anyhow!("Session builder failed for {}: {}", name, e))?
                }
            }
        };

        let builder = builder.with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("Optimization level failed for {}: {}", name, e))?;

        let session = builder.commit_from_memory(model_bytes)
            .map_err(|e| anyhow::anyhow!("Model load failed for {}: {}", name, e))?;

        for (i, input) in session.inputs().iter().enumerate() {
            log::info!("  {} input[{}]: {}", name, i, input.name());
        }
        for (i, output) in session.outputs().iter().enumerate() {
            log::info!("  {} output[{}]: {}", name, i, output.name());
        }

        log::info!("Successfully created {} session", name);
        Ok(session)
    }
}

/// Per-stream processor with independent state
/// Each stream has its own DFState and buffers, but shares ONNX sessions
pub struct StreamProcessor {
    /// Reference to shared ONNX sessions
    sessions: Arc<SharedSessions>,

    /// Stream identifier
    pub stream_id: u32,

    /// Whether this stream is enabled for processing
    enabled: AtomicBool,

    /// Per-stream signal processing state
    df_state: DFState,

    /// Spectrum buffer for FFT output [n_freqs] complex
    spec: Vec<Complex32>,

    /// ERB features buffer [nb_erb]
    erb_buf: Vec<f32>,

    /// Complex features buffer for DF [nb_df * 2] (real/imag interleaved)
    cplx_buf: Vec<f32>,

    /// Gains from ERB decoder [nb_erb]
    gains: Vec<f32>,

    /// DF coefficients from DF decoder [nb_df * df_order * 2]
    df_coefs: Vec<f32>,

    /// Rolling buffer for DF temporal filtering [df_order, nb_df] complex
    df_buf: Vec<Complex32>,
    df_buf_idx: usize,

    /// Encoder skip connection outputs (shape, data) pairs
    e0: (Vec<usize>, Vec<f32>),
    e1: (Vec<usize>, Vec<f32>),
    e2: (Vec<usize>, Vec<f32>),
    e3: (Vec<usize>, Vec<f32>),
    emb: (Vec<usize>, Vec<f32>),
    c0_enc: (Vec<usize>, Vec<f32>),
    lsnr: (Vec<usize>, Vec<f32>),

    /// Alpha for running normalization
    alpha: f32,

    /// Frame counter for logging
    frame_count: u64,
}

impl StreamProcessor {
    /// Create a new stream processor from shared sessions
    pub fn new(sessions: Arc<SharedSessions>, stream_id: u32) -> Result<Self> {
        // Clone config values before moving sessions
        let sr = sessions.config.sr;
        let fft_size = sessions.config.fft_size;
        let hop_size = sessions.config.hop_size;
        let nb_erb = sessions.config.nb_erb;
        let nb_df = sessions.config.nb_df;
        let min_nb_erb_freqs = sessions.config.min_nb_erb_freqs;
        let df_order = sessions.config.df_order;
        let alpha = sessions.config.alpha;

        // Initialize per-stream signal processing state
        let mut df_state = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
        df_state.init_norm_states(nb_df);

        // Allocate per-stream buffers
        let n_freqs = fft_size / 2 + 1;
        let spec = vec![Complex32::new(0.0, 0.0); n_freqs];
        let erb_buf = vec![0.0f32; nb_erb];
        let cplx_buf = vec![0.0f32; nb_df * 2];
        let gains = vec![0.0f32; nb_erb];
        let df_coefs = vec![0.0f32; nb_df * df_order * 2];
        let df_buf = vec![Complex32::new(0.0, 0.0); df_order * nb_df];

        let empty_tensor = (Vec::new(), Vec::new());

        log::info!(
            "Created StreamProcessor {}: n_freqs={}, nb_erb={}, nb_df={}, df_order={}",
            stream_id, n_freqs, nb_erb, nb_df, df_order
        );

        Ok(Self {
            sessions,
            stream_id,
            enabled: AtomicBool::new(true),
            df_state,
            spec,
            erb_buf,
            cplx_buf,
            gains,
            df_coefs,
            df_buf,
            df_buf_idx: 0,
            e0: empty_tensor.clone(),
            e1: empty_tensor.clone(),
            e2: empty_tensor.clone(),
            e3: empty_tensor.clone(),
            emb: empty_tensor.clone(),
            c0_enc: empty_tensor.clone(),
            lsnr: empty_tensor,
            alpha,
            frame_count: 0,
        })
    }

    /// Enable or disable this stream
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
        log::info!("Stream {} enabled: {}", self.stream_id, enabled);
    }

    /// Check if this stream is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Process a single frame of audio
    /// Returns attenuation in dB if processing, 0.0 if passthrough
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32> {
        let hop_size = self.sessions.config.hop_size;
        let fft_size = self.sessions.config.fft_size;
        let nb_erb = self.sessions.config.nb_erb;
        let nb_df = self.sessions.config.nb_df;

        // Log sizes on first call for debugging
        if self.frame_count == 0 {
            log::info!(
                "Stream {} init: hop_size={}, fft_size={}, nb_erb={}, nb_df={}, spec.len={}, erb_buf.len={}, cplx_buf.len={}",
                self.stream_id, hop_size, fft_size, nb_erb, nb_df,
                self.spec.len(), self.erb_buf.len(), self.cplx_buf.len()
            );
        }

        if input.len() != hop_size || output.len() != hop_size {
            bail!(
                "Invalid frame size: input={}, output={}, expected={}",
                input.len(), output.len(), hop_size
            );
        }

        // If disabled, passthrough
        if !self.is_enabled() {
            output.copy_from_slice(input);
            return Ok(0.0);
        }

        self.frame_count += 1;

        // Wrap ALL df_state operations in panic protection
        let process_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.process_frame_inner(input)
        }));

        match process_result {
            Ok(Ok(processed_output)) => {
                output.copy_from_slice(&processed_output);

                // Calculate attenuation for metrics
                let input_energy: f32 = input.iter().map(|x| x * x).sum();
                let output_energy: f32 = output.iter().map(|x| x * x).sum();
                let attenuation_db = if input_energy > 1e-10 {
                    10.0 * (output_energy / input_energy).log10()
                } else {
                    0.0
                };

                if self.frame_count <= 3 || self.frame_count % 500 == 0 {
                    log::info!(
                        "Stream {} frame {}: atten={:.1}dB, gain_avg={:.3}",
                        self.stream_id,
                        self.frame_count,
                        attenuation_db,
                        self.gains.iter().sum::<f32>() / self.gains.len() as f32
                    );
                }
                Ok(attenuation_db)
            }
            Ok(Err(e)) => {
                log::error!("Stream {} frame {} processing error: {}", self.stream_id, self.frame_count, e);
                output.copy_from_slice(input);
                Ok(0.0)
            }
            Err(panic_info) => {
                // Extract panic message
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    format!("{:?}", panic_info)
                };
                log::error!(
                    "Stream {} frame {} PANIC: {} | spec.len={}, erb.len={}, cplx.len={}, gains.len={}",
                    self.stream_id, self.frame_count, panic_msg,
                    self.spec.len(), self.erb_buf.len(), self.cplx_buf.len(), self.gains.len()
                );
                output.copy_from_slice(input);
                Ok(0.0)
            }
        }
    }

    /// Inner processing function that can panic - wrapped by process_frame
    fn process_frame_inner(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let hop_size = self.sessions.config.hop_size;
        let nb_df = self.sessions.config.nb_df;

        // Step 1: FFT analysis - ALWAYS run to maintain overlap-add state
        self.df_state.analysis(input, &mut self.spec);

        // Step 2: Extract ERB features (for normalization warmup)
        self.df_state.feat_erb(&self.spec, self.alpha, &mut self.erb_buf);

        // Step 3: Extract complex features for DF (for normalization warmup)
        self.df_state.feat_cplx_t(&self.spec[..nb_df], self.alpha, &mut self.cplx_buf);

        // Warmup period: skip inference but still run synthesis for overlap-add continuity
        let run_inference = self.frame_count > 10;

        if run_inference {
            // Check for NaN/Inf before inference
            let has_nan = self.erb_buf.iter().any(|x| !x.is_finite())
                || self.cplx_buf.iter().any(|x| !x.is_finite());

            if !has_nan {
                // Run ONNX inference
                self.run_inference()?;

                // Apply ERB gains to spectrum
                self.df_state.apply_mask(&mut self.spec, &self.gains);

                // Apply DF filter to low frequencies
                self.apply_df_filter();
            } else if self.frame_count % 100 == 0 {
                log::warn!("Stream {} frame {}: NaN/Inf in buffers, skipping inference",
                    self.stream_id, self.frame_count);
            }
        } else if self.frame_count == 1 || self.frame_count == 10 {
            log::info!("Stream {} frame {}: warmup (no inference)", self.stream_id, self.frame_count);
        }

        // ALWAYS run synthesis to maintain overlap-add state
        let mut output = vec![0.0f32; hop_size];
        self.df_state.synthesis(&mut self.spec, &mut output);

        Ok(output)
    }

    /// Run ONNX inference: encoder → ERB decoder → DF decoder
    fn run_inference(&mut self) -> Result<()> {
        let config = &self.sessions.config;
        let nb_erb = config.nb_erb;
        let nb_df = config.nb_df;

        // Create feat_erb tensor: [1, 1, 1, nb_erb]
        let erb_tensor = Tensor::from_array((
            [1usize, 1, 1, nb_erb],
            self.erb_buf.clone(),
        )).context("Failed to create erb_tensor")?;

        // Create feat_spec tensor: [1, 2, 1, nb_df]
        // cplx_buf is interleaved [r0,i0,r1,i1,...], reshape to [real_channel, imag_channel]
        let mut spec_data = vec![0.0f32; 2 * nb_df];
        for i in 0..nb_df {
            spec_data[i] = self.cplx_buf[i * 2];             // real -> channel 0
            spec_data[nb_df + i] = self.cplx_buf[i * 2 + 1]; // imag -> channel 1
        }
        let spec_tensor = Tensor::from_array((
            [1usize, 2, 1, nb_df],
            spec_data,
        )).context("Failed to create spec_tensor")?;

        // Run encoder
        {
            let mut enc_session = self.sessions.enc_session.lock();
            let enc_outputs = enc_session.run(ort::inputs![
                "feat_erb" => erb_tensor,
                "feat_spec" => spec_tensor,
            ]).context("Encoder run failed")?;

            fn extract_with_shape(value: &Value) -> Result<(Vec<usize>, Vec<f32>)> {
                let (shape_ref, data_ref) = value.try_extract_tensor::<f32>()?;
                let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();
                Ok((shape, data_ref.to_vec()))
            }

            self.e0 = extract_with_shape(&enc_outputs["e0"])?;
            self.e1 = extract_with_shape(&enc_outputs["e1"])?;
            self.e2 = extract_with_shape(&enc_outputs["e2"])?;
            self.e3 = extract_with_shape(&enc_outputs["e3"])?;
            self.emb = extract_with_shape(&enc_outputs["emb"])?;
            self.c0_enc = extract_with_shape(&enc_outputs["c0"])?;
            self.lsnr = extract_with_shape(&enc_outputs["lsnr"])?;
        }

        // Run ERB decoder: inputs=(emb, e3, e2, e1, e0)
        let emb_tensor = self.create_tensor_from_stored(&self.emb)?;
        let e3_tensor = self.create_tensor_from_stored(&self.e3)?;
        let e2_tensor = self.create_tensor_from_stored(&self.e2)?;
        let e1_tensor = self.create_tensor_from_stored(&self.e1)?;
        let e0_tensor = self.create_tensor_from_stored(&self.e0)?;

        {
            let mut erb_dec_session = self.sessions.erb_dec_session.lock();
            let erb_dec_outputs = erb_dec_session.run(ort::inputs![
                "emb" => emb_tensor,
                "e3" => e3_tensor,
                "e2" => e2_tensor,
                "e1" => e1_tensor,
                "e0" => e0_tensor,
            ])?;

            let (_, mask_data) = erb_dec_outputs["m"].try_extract_tensor::<f32>()?;
            for (i, &g) in mask_data.iter().enumerate() {
                if i < self.gains.len() {
                    self.gains[i] = g;
                }
            }
        }

        // Run DF decoder: inputs=(emb, c0)
        let emb_tensor2 = self.create_tensor_from_stored(&self.emb)?;
        let c0_tensor = self.create_tensor_from_stored(&self.c0_enc)?;

        {
            let mut df_dec_session = self.sessions.df_dec_session.lock();
            let df_dec_outputs = df_dec_session.run(ort::inputs![
                "emb" => emb_tensor2,
                "c0" => c0_tensor,
            ])?;

            let (_, coefs_data) = df_dec_outputs["coefs"].try_extract_tensor::<f32>()?;
            for (i, &c) in coefs_data.iter().enumerate() {
                if i < self.df_coefs.len() {
                    self.df_coefs[i] = c;
                }
            }
        }

        Ok(())
    }

    /// Create a tensor from stored (shape, data) pair
    fn create_tensor_from_stored(&self, stored: &(Vec<usize>, Vec<f32>)) -> Result<Tensor<f32>> {
        let (shape, data) = stored;
        match shape.len() {
            1 => Ok(Tensor::from_array(([shape[0]], data.clone()))?),
            2 => Ok(Tensor::from_array(([shape[0], shape[1]], data.clone()))?),
            3 => Ok(Tensor::from_array(([shape[0], shape[1], shape[2]], data.clone()))?),
            4 => Ok(Tensor::from_array(([shape[0], shape[1], shape[2], shape[3]], data.clone()))?),
            _ => bail!("Unsupported tensor rank: {}", shape.len()),
        }
    }

    /// Apply deep filtering to low frequency bins
    fn apply_df_filter(&mut self) {
        let config = &self.sessions.config;
        let nb_df = config.nb_df;
        let df_order = config.df_order;

        // Store current frame in rolling buffer
        for i in 0..nb_df {
            let idx = self.df_buf_idx * nb_df + i;
            if idx < self.df_buf.len() && i < self.spec.len() {
                self.df_buf[idx] = self.spec[i];
            }
        }

        // Apply DF filter: convolution with coefficients
        for i in 0..nb_df {
            if i >= self.spec.len() {
                break;
            }

            let mut acc = Complex32::new(0.0, 0.0);

            for t in 0..df_order {
                let buf_t = (self.df_buf_idx + df_order - t) % df_order;
                let buf_idx = buf_t * nb_df + i;

                if buf_idx >= self.df_buf.len() {
                    continue;
                }

                let coef_idx = (t * nb_df + i) * 2;
                if coef_idx + 1 >= self.df_coefs.len() {
                    continue;
                }

                let coef = Complex32::new(
                    self.df_coefs[coef_idx],
                    self.df_coefs[coef_idx + 1],
                );

                acc += coef * self.df_buf[buf_idx];
            }

            self.spec[i] = acc;
        }

        self.df_buf_idx = (self.df_buf_idx + 1) % df_order;
    }

    /// Reset the stream state
    pub fn reset(&mut self) {
        let config = &self.sessions.config;

        self.df_state = DFState::new(
            config.sr,
            config.fft_size,
            config.hop_size,
            config.nb_erb,
            config.min_nb_erb_freqs,
        );
        self.df_state.init_norm_states(config.nb_df);

        for s in &mut self.spec {
            *s = Complex32::new(0.0, 0.0);
        }
        self.erb_buf.fill(0.0);
        self.cplx_buf.fill(0.0);
        self.gains.fill(0.0);
        self.df_coefs.fill(0.0);
        for s in &mut self.df_buf {
            *s = Complex32::new(0.0, 0.0);
        }
        self.df_buf_idx = 0;

        self.e0 = (Vec::new(), Vec::new());
        self.e1 = (Vec::new(), Vec::new());
        self.e2 = (Vec::new(), Vec::new());
        self.e3 = (Vec::new(), Vec::new());
        self.emb = (Vec::new(), Vec::new());
        self.c0_enc = (Vec::new(), Vec::new());
        self.lsnr = (Vec::new(), Vec::new());

        self.frame_count = 0;

        log::info!("Stream {} state reset", self.stream_id);
    }

    pub fn sample_rate(&self) -> usize {
        self.sessions.config.sr
    }

    pub fn hop_size(&self) -> usize {
        self.sessions.config.hop_size
    }
}

/// Calculate normalization alpha from time constant
fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    (-dt / tau).exp()
}
