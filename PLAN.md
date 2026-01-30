# DeepFilterNet2 Flutter Test App - Implementation Plan

## Overview
Create a Flutter app with Rust audio core to test DeepFilterNet noise suppression on Android.
Audio recording and playback handled entirely in Rust (no JNI for audio).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flutter UI (Dart)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────────┐  │
│  │ Record  │  │  Stop   │  │  Play   │  │ Status Display │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────────────┘  │
│       │            │            │                            │
└───────┼────────────┼────────────┼────────────────────────────┘
        │            │            │
        └────────────┼────────────┘
                     │ flutter_rust_bridge (FFI)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Audio Core                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    AudioEngine                        │   │
│  │  ┌─────────┐  ┌──────────────┐  ┌─────────────────┐  │   │
│  │  │  Oboe   │──▶│ DeepFilter  │──▶│   WAV Writer   │  │   │
│  │  │(record) │  │ Net (denoise)│  │  (hound crate) │  │   │
│  │  └─────────┘  └──────────────┘  └─────────────────┘  │   │
│  │                                                       │   │
│  │  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │   WAV Reader    │──▶│   Oboe (playback)       │   │   │
│  │  │  (hound crate)  │  │                          │   │   │
│  │  └─────────────────┘  └──────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
deepfilter_test/
├── flutter_app/                    # Flutter project
│   ├── lib/
│   │   ├── main.dart              # App entry point
│   │   ├── src/
│   │   │   └── rust/              # Generated flutter_rust_bridge bindings
│   │   └── ui/
│   │       └── home_screen.dart   # Main UI with record/play controls
│   ├── android/
│   │   └── app/
│   │       └── build.gradle.kts   # Android config with native libs
│   └── pubspec.yaml
│
├── rust/                           # Rust native library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                 # FFI exports for flutter_rust_bridge
│       ├── api/
│       │   └── audio_engine.rs    # Public API for Flutter
│       ├── audio/
│       │   ├── mod.rs
│       │   ├── recorder.rs        # Oboe-based audio recording
│       │   └── player.rs          # Oboe-based audio playback
│       ├── processing/
│       │   ├── mod.rs
│       │   └── deepfilter.rs      # DeepFilterNet wrapper
│       └── io/
│           ├── mod.rs
│           └── wav.rs             # WAV file read/write
│
├── models/                         # DeepFilterNet model files
│   └── DeepFilterNet3_onnx_mobile.tar.gz
│
└── PLAN.md                        # This file
```

## Dependencies

### Rust (Cargo.toml)
```toml
[package]
name = "deepfilter_audio"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "staticlib"]

[dependencies]
# Flutter-Rust bridge
flutter_rust_bridge = "2"

# Audio I/O (Oboe for Android)
oboe = { version = "0.6", features = ["java-interface"] }

# DeepFilterNet - use git dependency for latest android support
df = { git = "https://github.com/KaleyraVideo/DeepFilterNet.git", package = "deep_filter", features = ["tract", "android"] }

# Audio file I/O
hound = "3.5"

# Utilities
anyhow = "1.0"
log = "0.4"
android_logger = "0.13"
parking_lot = "0.12"

[target.'cfg(target_os = "android")'.dependencies]
jni = "0.21"  # Required by oboe for audio focus

[profile.release]
lto = true
opt-level = 3
```

### Flutter (pubspec.yaml)
```yaml
dependencies:
  flutter:
    sdk: flutter
  flutter_rust_bridge: ^2.0.0
  permission_handler: ^11.0.0
  path_provider: ^2.1.0
```

## Implementation Steps

### Phase 1: Project Setup
1. Create Flutter project in `deepfilter_test/flutter_app`
2. Create Rust workspace in `deepfilter_test/rust`
3. Initialize flutter_rust_bridge integration
4. Set up Android NDK build configuration

### Phase 2: Rust Audio Core
1. **Recorder module** (`audio/recorder.rs`)
   - Initialize Oboe audio input stream
   - Configure: 48kHz, mono, f32 format
   - Callback-based recording to ring buffer
   - Convert to 480-sample frames for DeepFilterNet

2. **DeepFilter wrapper** (`processing/deepfilter.rs`)
   - Load DeepFilterNet3 mobile model from assets
   - Process 480-sample frames (10ms at 48kHz)
   - Return denoised audio

3. **WAV I/O** (`io/wav.rs`)
   - Write 48kHz mono 16-bit WAV files
   - Read WAV files for playback

4. **Player module** (`audio/player.rs`)
   - Initialize Oboe audio output stream
   - Play back WAV file content

5. **API layer** (`api/audio_engine.rs`)
   - `start_recording(output_path: String) -> Result<()>`
   - `stop_recording() -> Result<()>`
   - `play_file(file_path: String) -> Result<()>`
   - `stop_playback() -> Result<()>`
   - `get_status() -> RecordingStatus`

### Phase 3: Flutter UI
1. Simple home screen with:
   - Record button (toggles recording)
   - Play button (plays last recorded file)
   - Status text (recording time, playback progress)
   - Permission handling for RECORD_AUDIO

### Phase 4: Integration & Build
1. Generate flutter_rust_bridge bindings
2. Configure Android Gradle for native libs
3. Bundle DeepFilterNet model in assets
4. Build and test on Android device

## Key Technical Details

### Audio Format
- Sample rate: 48000 Hz (required by DeepFilterNet)
- Channels: 1 (mono)
- Frame size: 480 samples (10ms) - DeepFilterNet hop size
- Bit depth: 16-bit for WAV, f32 internally

### DeepFilterNet Integration
```rust
use df::tract::{DfParams, DfTract, RuntimeParams};

pub struct DeepFilter {
    df: DfTract,
}

impl DeepFilter {
    pub fn new(model_bytes: &[u8]) -> Result<Self> {
        let params = DfParams::from_bytes(model_bytes)?;
        let runtime = RuntimeParams::default_with_ch(1).with_atten_lim(100.0);
        let df = DfTract::new(params, &runtime)?;
        Ok(Self { df })
    }

    pub fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> f32 {
        let input = ndarray::Array2::from_shape_vec((1, 480), input.to_vec()).unwrap();
        let mut output_arr = ndarray::Array2::zeros((1, 480));
        let lsnr = self.df.process(input.view(), output_arr.view_mut()).unwrap();
        output.copy_from_slice(output_arr.as_slice().unwrap());
        lsnr
    }
}
```

### Oboe Recording
```rust
use oboe::{AudioStream, AudioStreamBuilder, DataCallbackResult, PerformanceMode};

struct RecordingCallback {
    buffer: Arc<Mutex<VecDeque<f32>>>,
}

impl oboe::AudioInputCallback for RecordingCallback {
    fn on_audio_ready(&mut self, stream: &mut dyn AudioInputStream, data: &[f32]) -> DataCallbackResult {
        self.buffer.lock().extend(data.iter().copied());
        DataCallbackResult::Continue
    }
}
```

## Build Commands

### Rust for Android
```bash
# Install cargo-ndk
cargo install cargo-ndk

# Add Android targets
rustup target add aarch64-linux-android armv7-linux-androideabi

# Build for Android
cd rust
cargo ndk -t arm64-v8a -t armeabi-v7a -o ../flutter_app/android/app/src/main/jniLibs build --release
```

### Flutter
```bash
cd flutter_app
flutter_rust_bridge_codegen generate
flutter build apk --release
```

## Files to Create

1. `rust/Cargo.toml` - Rust dependencies
2. `rust/src/lib.rs` - Main lib with FFI exports
3. `rust/src/api/audio_engine.rs` - Flutter API
4. `rust/src/audio/mod.rs` - Audio module
5. `rust/src/audio/recorder.rs` - Oboe recorder
6. `rust/src/audio/player.rs` - Oboe player
7. `rust/src/processing/mod.rs` - Processing module
8. `rust/src/processing/deepfilter.rs` - DeepFilter wrapper
9. `rust/src/io/mod.rs` - I/O module
10. `rust/src/io/wav.rs` - WAV read/write
11. `flutter_app/lib/main.dart` - Flutter app
12. `flutter_app/lib/ui/home_screen.dart` - UI
13. `flutter_app/pubspec.yaml` - Flutter deps
14. `flutter_app/android/app/build.gradle.kts` - Android config
