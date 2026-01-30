use anyhow::{Context, Result, bail};
use std::io::{Cursor, Read};
use std::sync::Once;
use flate2::read::GzDecoder;
use tar::Archive;
use ini::Ini;  // from rust-ini crate
use ndarray::{Array, ArrayView2, ArrayViewMut2, Axis, s};
use ort::session::{Session, builder::GraphOptimizationLevel};

use crate::audio::FRAME_SIZE;
use df::DFState;

// Initialize ORT once
static ORT_INIT: Once = Once::new();

/// Initialize ONNX Runtime (must be called before creating sessions)
fn init_ort() -> Result<()> {
    let mut init_result: Result<()> = Ok(());

    ORT_INIT.call_once(|| {
        log::info!("Initializing ONNX Runtime...");

        // On Android, libonnxruntime.so is loaded from the app's jniLibs
        // Need to use full library name for dlopen
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
            // On desktop, try to init with default settings
            match ort::init() {
                Ok(_) => log::info!("ONNX Runtime initialized successfully"),
                Err(e) => {
                    log::error!("Failed to initialize ONNX Runtime: {}", e);
                    init_result = Err(anyhow::anyhow!("ORT init failed: {}", e));
                }
            }
        }
    });

    init_result
}

/// DeepFilter model configuration parsed from config.ini
struct DfConfig {
    sr: usize,
    hop_size: usize,
    fft_size: usize,
    nb_erb: usize,
    nb_df: usize,
    min_nb_erb_freqs: usize,
    df_order: usize,
    conv_lookahead: usize,
    df_lookahead: usize,
    alpha: f32,
}

/// ONNX Runtime based DeepFilter for noise suppression
/// Uses NNAPI on Android for GPU/NPU acceleration
pub struct DeepFilter {
    // ONNX Runtime sessions
    enc_session: Session,
    erb_dec_session: Session,
    df_dec_session: Session,

    // Signal processing state
    df_state: DFState,

    // Configuration
    config: DfConfig,

    // Buffers
    erb_buf: Vec<f32>,
    cplx_buf: Vec<f32>,
    spec_buf: Vec<f32>,

    // Hidden states for recurrent models
    enc_hidden: Vec<f32>,
    erb_dec_hidden: Vec<f32>,
    df_dec_hidden: Vec<f32>,
}

impl DeepFilter {
    /// Create a new DeepFilter instance from model bytes (tar.gz archive)
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
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
            let path_str = path.to_string_lossy();

            // Get filename and skip macOS metadata files (start with ._)
            let filename = path.file_name()
                .map(|f| f.to_string_lossy())
                .unwrap_or_default();

            if filename.starts_with("._") {
                continue;  // Skip macOS extended attribute files
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
                // Convert bytes to string, replacing invalid UTF-8
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

        // Calculate alpha for normalization
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
            "DeepFilter config: sr={}, hop_size={}, fft_size={}, nb_erb={}, nb_df={}",
            sr, hop_size, fft_size, nb_erb, nb_df
        );

        // Create ONNX Runtime sessions
        let enc_session = Self::create_session(&enc_bytes, "encoder")?;
        let erb_dec_session = Self::create_session(&erb_dec_bytes, "erb_decoder")?;
        let df_dec_session = Self::create_session(&df_dec_bytes, "df_decoder")?;

        // Initialize signal processing state
        let mut df_state = DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs);
        df_state.init_norm_states(nb_df);

        // Allocate buffers
        let n_freqs = fft_size / 2 + 1;
        let erb_buf = vec![0.0f32; nb_erb];
        let cplx_buf = vec![0.0f32; nb_df * 2];
        let spec_buf = vec![0.0f32; n_freqs * 2];

        // Initialize hidden states (sizes depend on model architecture)
        // These are typically determined by examining the ONNX model inputs
        let enc_hidden = vec![0.0f32; 512];  // Adjust based on model
        let erb_dec_hidden = vec![0.0f32; 512];
        let df_dec_hidden = vec![0.0f32; 512];

        log::info!("DeepFilter initialized with ONNX Runtime");

        Ok(Self {
            enc_session,
            erb_dec_session,
            df_dec_session,
            df_state,
            config,
            erb_buf,
            cplx_buf,
            spec_buf,
            enc_hidden,
            erb_dec_hidden,
            df_dec_hidden,
        })
    }

    /// Create an ONNX Runtime session with optimal settings
    fn create_session(model_bytes: &[u8], name: &str) -> Result<Session> {
        log::info!("Creating ONNX session for {} ({} bytes)", name, model_bytes.len());

        let builder = match Session::builder() {
            Ok(b) => b,
            Err(e) => {
                log::error!("Failed to create session builder for {}: {:?}", name, e);
                return Err(anyhow::anyhow!("Session builder failed: {}", e));
            }
        };

        let builder = match builder.with_optimization_level(GraphOptimizationLevel::Level1) {
            Ok(b) => b,
            Err(e) => {
                log::error!("Failed to set optimization level for {}: {:?}", name, e);
                return Err(anyhow::anyhow!("Optimization level failed: {}", e));
            }
        };

        let session = match builder.commit_from_memory(model_bytes) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to load {} model: {:?}", name, e);
                return Err(anyhow::anyhow!("Model load failed for {}: {}", name, e));
            }
        };

        // Log input/output info
        for (i, input) in session.inputs().iter().enumerate() {
            log::info!("  {} input[{}]: {}", name, i, input.name());
        }
        for (i, output) in session.outputs().iter().enumerate() {
            log::info!("  {} output[{}]: {}", name, i, output.name());
        }

        log::info!("Successfully created {} session", name);
        Ok(session)
    }

    /// Process a single frame of audio
    /// Input and output must be FRAME_SIZE samples
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32> {
        if input.len() != FRAME_SIZE || output.len() != FRAME_SIZE {
            bail!(
                "Invalid frame size: input={}, output={}, expected={}",
                input.len(), output.len(), FRAME_SIZE
            );
        }

        // For now, use a simplified processing pipeline
        // TODO: Implement full DeepFilter processing with ort

        // Step 1: Apply FFT to get spectrum
        let mut spec = vec![0.0f32; self.config.fft_size / 2 + 1];

        // Simple passthrough for now until full implementation
        // The actual implementation requires careful handling of:
        // - FFT/IFFT using DFState
        // - ERB feature extraction
        // - Running encoder/decoder networks
        // - Applying gains and DF coefficients

        output.copy_from_slice(input);

        // Return dummy LSNR
        Ok(0.0)
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.config.sr
    }

    /// Get the hop size (frame size)
    pub fn hop_size(&self) -> usize {
        self.config.hop_size
    }
}

/// Calculate normalization alpha from time constant
fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    (-dt / tau).exp()
}
