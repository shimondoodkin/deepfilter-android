# DeepFilterNet Test App

Flutter app with Rust audio core to test DeepFilterNet3 noise suppression on Android.

## Project Structure

```
deepfilter_test/
├── flutter_app/           # Flutter project
│   ├── lib/main.dart      # Main app with record/play UI
│   ├── assets/            # DeepFilterNet3 model
│   ├── rust_builder/      # FRB Rust wrapper (bridges to deepfilter_audio)
│   └── android/app/src/main/jniLibs/  # Compiled .so files
├── rust/                  # Rust audio core (deepfilter_audio crate)
│   └── src/
│       ├── api/           # Flutter API (audio_engine.rs)
│       ├── audio/         # Audio recording/playback (Oboe for Android)
│       ├── processing/    # DeepFilterNet wrapper
│       └── io/            # WAV file I/O
└── models/                # Model files
```

## Prerequisites

1. **Flutter SDK** - Install via snap: `sudo snap install flutter --classic`
2. **Rust** - Install via rustup: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. **Android SDK + NDK** - Required for Android builds
4. **cargo-ndk** - For cross-compiling to Android

### Install Android SDK/NDK

```bash
# Install via Android Studio or command line
# After installing, set environment variables:
export ANDROID_SDK_ROOT=/opt/android-sdk
export ANDROID_NDK_HOME=/opt/android-sdk/ndk/27.0.12077973
```

### Install Rust Android Targets

```bash
source ~/.cargo/env
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android
cargo install cargo-ndk
```

## Build Commands

### Quick Build (ARM64 only)

```bash
# 1. Build Rust library for Android
source ~/.cargo/env
export ANDROID_NDK_HOME=/opt/android-sdk/ndk/27.0.12077973
cd flutter_app/rust_builder
cargo ndk -t arm64-v8a build --release

# 2. Copy .so to jniLibs
cp target/aarch64-linux-android/release/librust_lib_deepfilter_test.so \
   ../android/app/src/main/jniLibs/arm64-v8a/

# 3. Build Flutter APK
cd ..
flutter build apk --debug
```

### Full Build (All architectures)

```bash
# Build for all Android architectures
cd flutter_app/rust_builder
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 build --release

# Copy all .so files
cp target/aarch64-linux-android/release/librust_lib_deepfilter_test.so ../android/app/src/main/jniLibs/arm64-v8a/
cp target/armv7-linux-androideabi/release/librust_lib_deepfilter_test.so ../android/app/src/main/jniLibs/armeabi-v7a/
cp target/x86_64-linux-android/release/librust_lib_deepfilter_test.so ../android/app/src/main/jniLibs/x86_64/
```

### Regenerate Flutter-Rust Bridge Bindings

Only needed if you modify the Rust API:

```bash
cargo install flutter_rust_bridge_codegen
cd flutter_app
flutter_rust_bridge_codegen generate
```

## One-Liner Build

```bash
source ~/.cargo/env && \
export ANDROID_NDK_HOME=/opt/android-sdk/ndk/27.0.12077973 && \
cd /path/to/deepfilter_test/flutter_app/rust_builder && \
cargo ndk -t arm64-v8a build --release && \
cp target/aarch64-linux-android/release/librust_lib_deepfilter_test.so ../android/app/src/main/jniLibs/arm64-v8a/ && \
cd .. && \
flutter build apk --debug
```

## How It Works

1. **Model Loading**: DeepFilterNet3 ONNX model loaded from Flutter assets on startup
2. **Recording**: Captures audio using Oboe (Android) with i16 format for device compatibility
3. **Processing**: After recording stops, audio is passed through DeepFilterNet3 for noise suppression
4. **Saving**: Processed audio is saved as a 48kHz mono WAV file
5. **Playback**: Plays back the denoised recording via Oboe output stream

## Audio Configuration

- Sample rate: 48000 Hz
- Channels: Mono (1)
- Frame size: 480 samples (10ms)
- Format: 16-bit PCM (i16) for Oboe, converted to f32 for processing
- Output: 16-bit PCM WAV files

## Architecture Notes

### Thread Safety

The Rust API uses global `Mutex<Option<T>>` storage for the recorder and player to handle Flutter-Rust bridge's potential cross-thread calls. The DeepFilter model itself is not thread-safe (uses `Rc` internally), so it's created fresh when needed for processing.

### Android-Specific

- Uses Oboe with `PerformanceMode::None` and `SharingMode::Shared` for better Samsung device compatibility
- Audio streams use i16 format (better device support) with f32 conversion in callbacks
- Stream lifecycle is managed via callback return values (`DataCallbackResult::Stop`)

## Troubleshooting

### "Recorder not found" error
This was caused by thread-local storage being inaccessible across threads. Fixed by using global Mutex storage.

### Build fails with "linker not found"
Ensure `ANDROID_NDK_HOME` is set correctly and the NDK is installed.

### Permission denied on recording
App requires `RECORD_AUDIO` permission. Check AndroidManifest.xml and grant permission in device settings.

## Files Modified During Development

Key files that were created or significantly modified:
- `rust/src/api/audio_engine.rs` - Main API with global state management
- `rust/src/audio/recorder.rs` - Oboe-based audio recording
- `rust/src/audio/player.rs` - Oboe-based audio playback
- `rust/src/processing/deepfilter.rs` - DeepFilterNet wrapper
- `flutter_app/lib/main.dart` - Flutter UI
- `flutter_app/rust_builder/src/api.rs` - FRB wrapper API
