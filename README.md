# DeepFilterNet Test App

Flutter app with Rust audio core to test DeepFilterNet3 noise suppression on Android with GPU/NPU acceleration via ONNX Runtime + NNAPI.

## Features

- **ONNX Runtime with NNAPI** - GPU/NPU acceleration on Android 8.1+
- DeepFilterNet3 neural network for noise suppression
- Oboe-based audio recording/playback for low latency
- Post-recording noise filtering with WAV output
- **Dual parallel stream processing** with runtime enable/disable control

## Project Structure

```
deepfilter_test/
├── flutter_app/           # Flutter project
│   ├── lib/main.dart      # Main app with record/play UI
│   ├── assets/            # DeepFilterNet3 model (.tar.gz)
│   ├── rust_builder/      # FRB Rust wrapper
│   └── android/app/src/main/jniLibs/  # Native libraries
│       └── arm64-v8a/
│           ├── librust_lib_deepfilter_test.so  # Rust code (2.5MB)
│           ├── libonnxruntime.so               # ONNX Runtime (19MB)
│           └── libc++_shared.so                # C++ runtime (1.8MB)
├── rust/                  # Rust audio core (deepfilter_audio crate)
│   └── src/
│       ├── api/           # Flutter API (audio_engine.rs)
│       ├── audio/         # Audio recording/playback (Oboe)
│       ├── processing/    # DeepFilter with ONNX Runtime
│       └── io/            # WAV file I/O
└── models/                # Model files
```

## Prerequisites

1. **Flutter SDK** - `sudo snap install flutter --classic`
2. **Rust** - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. **Android SDK + NDK 27+**
4. **cargo-ndk** - `cargo install cargo-ndk`

### Install Android Targets

```bash
source ~/.cargo/env
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
```

## Build Commands

### 1. Download ONNX Runtime for Android (one-time setup)

**Important:** ort crate 2.0.0-rc.11 requires ONNX Runtime 1.23.x

```bash
# Download from Maven Central
curl -sL "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.23.0/onnxruntime-android-1.23.0.aar" -o /tmp/onnxruntime.aar

# Extract .so files
cd /tmp && unzip -o onnxruntime.aar "jni/*"

# Copy to jniLibs (for each architecture you need)
cp jni/arm64-v8a/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/arm64-v8a/
cp jni/armeabi-v7a/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/armeabi-v7a/
cp jni/x86_64/libonnxruntime.so /path/to/flutter_app/android/app/src/main/jniLibs/x86_64/
```

### 2. Build Rust Library

```bash
source ~/.cargo/env
export ANDROID_NDK_HOME=/opt/android-sdk/ndk/27.0.12077973

cd flutter_app/rust_builder

# ARM64 only (most devices)
cargo ndk -t arm64-v8a build --release

# Copy to jniLibs
cp target/aarch64-linux-android/release/librust_lib_deepfilter_test.so \
   ../android/app/src/main/jniLibs/arm64-v8a/
```

### 3. Build Flutter APK

```bash
cd flutter_app
flutter build apk --debug
```

### One-Liner Build

```bash
source ~/.cargo/env && \
export ANDROID_NDK_HOME=/opt/android-sdk/ndk/27.0.12077973 && \
cd flutter_app/rust_builder && \
cargo ndk -t arm64-v8a build --release && \
cp target/aarch64-linux-android/release/librust_lib_deepfilter_test.so \
   ../android/app/src/main/jniLibs/arm64-v8a/ && \
cd .. && flutter build apk --debug
```

## Architecture

### ONNX Runtime Integration

The app uses ONNX Runtime with dynamic loading (`load-dynamic` feature):

1. **libonnxruntime.so** (19MB) is downloaded from Maven and included in jniLibs
2. At runtime, `ort::init_from("libonnxruntime.so")` loads the library
3. ONNX Runtime automatically uses NNAPI if available (GPU/NPU acceleration)

**Version compatibility:**
| ort crate | ONNX Runtime |
|-----------|--------------|
| 2.0.0-rc.11 | 1.23.x |
| 1.16.x | 1.16.x |

### DeepFilterNet3 Pipeline

The model archive contains three ONNX models:
- `enc.onnx` - Encoder network
- `erb_dec.onnx` - ERB (Equivalent Rectangular Bandwidth) decoder
- `df_dec.onnx` - Deep Filtering decoder

Processing flow:
1. FFT → ERB features → Encoder
2. ERB Decoder → Gain mask
3. DF Decoder → Filter coefficients
4. Apply gains and DF → IFFT → Output

### Dual Stream Architecture

The app supports **two parallel processing streams** (A and B) that share the same ONNX Runtime sessions but maintain independent state:

```
                    ┌─────────────────────────────────────────┐
                    │           SharedSessions                │
                    │  (Arc-wrapped, thread-safe ONNX Sessions)│
                    │  - enc_session                          │
                    │  - erb_dec_session                      │
                    │  - df_dec_session                       │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌──────────────────┐ ┌──────────────────┐
    │ StreamProcessor A│ │ StreamProcessor B│   ... (more streams possible)
    │ - stream_id: 0   │ │ - stream_id: 1   │
    │ - enabled: bool  │ │ - enabled: bool  │
    │ - df_state       │ │ - df_state       │
    │ - buffers        │ │ - buffers        │
    │ - hidden states  │ │ - hidden states  │
    └──────────────────┘ └──────────────────┘
```

**API Functions:**
```dart
// Enable/disable streams at runtime
await setStreamAEnabled(enabled: true);
await setStreamBEnabled(enabled: false);

// Check stream status
bool aEnabled = await isStreamAEnabled();
bool bEnabled = await isStreamBEnabled();

// Status includes stream states
RecordingStatus status = await getStatus();
print(status.streamAEnabled);  // true/false
print(status.streamBEnabled);  // true/false
```

**Processing behavior:**
- Both streams process the **same input** (mic audio)
- Final output is the **average** of enabled streams
- Disabled streams output passthrough (unprocessed audio)
- Each stream maintains independent recurrent hidden states

This enables A/B testing of DeepFilter with instant switching.

### Thread Safety

- Global `Mutex<Option<T>>` storage for recorder/player
- ONNX Runtime sessions are thread-safe via `Arc<SharedSessions>`
- `unsafe impl Send + Sync` for audio types (Oboe/cpal streams)
- Per-stream state isolation prevents interference

## Audio Configuration

- Sample rate: 48000 Hz
- Channels: Mono (1)
- Frame size: 480 samples (10ms)
- Format: 16-bit PCM (i16) for Oboe, f32 internally

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ort` | ONNX Runtime Rust bindings (with NNAPI) |
| `df` | DeepFilterNet signal processing |
| `oboe` | Android audio I/O |
| `flutter_rust_bridge` | Dart-Rust FFI |

## Cargo.toml Configuration

```toml
# ONNX Runtime with dynamic loading
ort = { version = "2.0.0-rc.9", features = ["ndarray", "load-dynamic"] }

# DeepFilterNet - transforms only (signal processing)
df = { features = ["transforms", "logging"] }
```

## Troubleshooting

### "dlopen failed" / "Failed to initialize ONNX Runtime"
- Ensure `libonnxruntime.so` is in jniLibs for your target architecture
- Use full library name: `ort::init_from("libonnxruntime.so")`

### "ort is not compatible with ONNX Runtime binary"
- Version mismatch - ort 2.0.0-rc.11 requires ONNX Runtime 1.23.x
- Download correct version from Maven

### "protobuf parsing failed"
- Check for macOS `._` metadata files in tar archive
- Code now skips files starting with `._`

### "Failed to parse config.ini" / "stream did not contain valid UTF-8"
- Fixed by reading to bytes first, then using `from_utf8_lossy()`

### "Recorder not found"
- Fixed by using global Mutex storage instead of thread_local

### NNAPI not being used
- NNAPI availability depends on device and Android version (8.1+)
- Check logcat for ONNX Runtime execution provider messages

## Development Notes

### Model Archive Format
The DeepFilterNet3 mobile model (`DeepFilterNet3_onnx_mobile.tar.gz`) contains:
- `config.ini` - Model configuration
- `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` - ONNX models
- `._*` files - macOS metadata (skipped during extraction)

### Key Files
- `rust/src/processing/deepfilter.rs` - ONNX Runtime session management
- `rust/src/api/audio_engine.rs` - Thread-safe Flutter API
- `rust/src/audio/recorder.rs` - Oboe audio recording
- `rust/src/audio/player.rs` - Oboe audio playback

## License

MIT
