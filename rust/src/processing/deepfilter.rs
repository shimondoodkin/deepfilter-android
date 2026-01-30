use anyhow::{Context, Result, bail};
use std::io::{Cursor, Read};
use std::sync::{Arc, Once};
use std::sync::atomic::{AtomicBool, Ordering};
use flate2::read::GzDecoder;
use tar::Archive;
use ini::Ini;
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
            // On desktop, ort::init() returns EnvironmentBuilder
            // We just need to call it to initialize
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
pub struct SharedSessions {
    pub enc_session: Session,
    pub erb_dec_session: Session,
    pub df_dec_session: Session,
    pub config: DfConfig,
}

// Session is Send + Sync, so SharedSessions can be shared
unsafe impl Send for SharedSessions {}
unsafe impl Sync for SharedSessions {}

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
            "DeepFilter config: sr={}, hop_size={}, fft_size={}, nb_erb={}, nb_df={}",
            sr, hop_size, fft_size, nb_erb, nb_df
        );

        // Create ONNX Runtime sessions
        let enc_session = Self::create_session(&enc_bytes, "encoder")?;
        let erb_dec_session = Self::create_session(&erb_dec_bytes, "erb_decoder")?;
        let df_dec_session = Self::create_session(&df_dec_bytes, "df_decoder")?;

        log::info!("SharedSessions initialized with ONNX Runtime");

        Ok(Arc::new(Self {
            enc_session,
            erb_dec_session,
            df_dec_session,
            config,
        }))
    }

    fn create_session(model_bytes: &[u8], name: &str) -> Result<Session> {
        log::info!("Creating ONNX session for {} ({} bytes)", name, model_bytes.len());

        let builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Session builder failed for {}: {}", name, e))?;

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

    /// Per-stream buffers
    erb_buf: Vec<f32>,
    cplx_buf: Vec<f32>,
    spec_buf: Vec<f32>,

    /// Per-stream hidden states for recurrent models
    enc_hidden: Vec<f32>,
    erb_dec_hidden: Vec<f32>,
    df_dec_hidden: Vec<f32>,
}

impl StreamProcessor {
    /// Create a new stream processor from shared sessions
    pub fn new(sessions: Arc<SharedSessions>, stream_id: u32) -> Result<Self> {
        let config = &sessions.config;

        // Initialize per-stream signal processing state
        let mut df_state = DFState::new(
            config.sr,
            config.fft_size,
            config.hop_size,
            config.nb_erb,
            config.min_nb_erb_freqs,
        );
        df_state.init_norm_states(config.nb_df);

        // Allocate per-stream buffers
        let n_freqs = config.fft_size / 2 + 1;
        let erb_buf = vec![0.0f32; config.nb_erb];
        let cplx_buf = vec![0.0f32; config.nb_df * 2];
        let spec_buf = vec![0.0f32; n_freqs * 2];

        // Initialize hidden states
        let enc_hidden = vec![0.0f32; 512];
        let erb_dec_hidden = vec![0.0f32; 512];
        let df_dec_hidden = vec![0.0f32; 512];

        log::info!("Created StreamProcessor {}", stream_id);

        Ok(Self {
            sessions,
            stream_id,
            enabled: AtomicBool::new(true),
            df_state,
            erb_buf,
            cplx_buf,
            spec_buf,
            enc_hidden,
            erb_dec_hidden,
            df_dec_hidden,
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
    /// Returns processed output if enabled, or copies input if disabled
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32> {
        if input.len() != FRAME_SIZE || output.len() != FRAME_SIZE {
            bail!(
                "Invalid frame size: input={}, output={}, expected={}",
                input.len(), output.len(), FRAME_SIZE
            );
        }

        // If disabled, just copy input to output (passthrough)
        if !self.is_enabled() {
            output.copy_from_slice(input);
            return Ok(0.0);
        }

        // TODO: Implement full DeepFilter processing with shared sessions
        // For now, passthrough
        output.copy_from_slice(input);
        Ok(0.0)
    }

    /// Reset the stream state (for starting a new recording)
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

        // Reset buffers
        self.erb_buf.fill(0.0);
        self.cplx_buf.fill(0.0);
        self.spec_buf.fill(0.0);
        self.enc_hidden.fill(0.0);
        self.erb_dec_hidden.fill(0.0);
        self.df_dec_hidden.fill(0.0);

        log::info!("Stream {} state reset", self.stream_id);
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.sessions.config.sr
    }

    /// Get the hop size (frame size)
    pub fn hop_size(&self) -> usize {
        self.sessions.config.hop_size
    }
}

/// Legacy DeepFilter struct for backwards compatibility
/// Wraps a single StreamProcessor
pub struct DeepFilter {
    processor: StreamProcessor,
}

impl DeepFilter {
    /// Create a new DeepFilter instance from model bytes
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        let sessions = SharedSessions::new(model_bytes)?;
        let processor = StreamProcessor::new(sessions, 0)?;
        Ok(Self { processor })
    }

    /// Process a single frame of audio
    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32> {
        self.processor.process_frame(input, output)
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.processor.sample_rate()
    }

    /// Get the hop size
    pub fn hop_size(&self) -> usize {
        self.processor.hop_size()
    }
}

/// Calculate normalization alpha from time constant
fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    (-dt / tau).exp()
}
