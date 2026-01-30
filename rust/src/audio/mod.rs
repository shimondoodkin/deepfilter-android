mod recorder;
mod player;

pub use recorder::AudioRecorder;
pub use player::AudioPlayer;

/// Audio configuration constants
pub const SAMPLE_RATE: u32 = 48000;
pub const CHANNELS: u16 = 1;
pub const FRAME_SIZE: usize = 480; // 10ms at 48kHz - DeepFilter hop size
