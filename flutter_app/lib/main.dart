import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';

import 'src/rust/api.dart';
import 'src/rust/frb_generated.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await RustLib.init();
  runApp(const DeepFilterApp());
}

class DeepFilterApp extends StatelessWidget {
  const DeepFilterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DeepFilter Test',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isInitialized = false;
  bool _isRecording = false;
  bool _isPlaying = false;
  String _statusText = 'Initializing...';
  String? _lastRecordedFile;
  int _recordingDuration = 0;
  Timer? _statusTimer;
  bool _streamAEnabled = true;
  bool _streamBEnabled = true;
  double _cpuUsage = -1;
  double _gpuUsage = -1;
  bool _nnapiAvailable = false;

  @override
  void initState() {
    super.initState();
    // Start metrics timer immediately
    _statusTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
      _updateMetrics();
    });
    _initialize();
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    super.dispose();
  }

  Future<void> _initialize() async {
    try {
      // Request microphone permission
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        setState(() {
          _statusText = 'Microphone permission denied';
        });
        return;
      }

      // Load DeepFilter model from assets
      setState(() {
        _statusText = 'Loading DeepFilter model...';
      });

      final modelData = await rootBundle.load('assets/DeepFilterNet3_onnx_mobile.tar.gz');
      await initEngine(modelData: modelData.buffer.asUint8List());

      setState(() {
        _isInitialized = true;
        _statusText = 'Ready to record';
      });

      // Switch to full status updates (includes recording duration)
      _statusTimer?.cancel();
      _statusTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _updateStatus();
      });
    } catch (e) {
      setState(() {
        _statusText = 'Initialization failed: $e';
      });
    }
  }

  Future<void> _startRecording() async {
    if (!_isInitialized || _isRecording) return;

    try {
      final dir = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final filePath = '${dir.path}/recording_$timestamp.wav';

      await startRecording(outputPath: filePath);

      setState(() {
        _isRecording = true;
        _lastRecordedFile = filePath;
        _recordingDuration = 0;
        _statusText = 'Recording...';
      });

      // Switch to faster updates during recording
      _statusTimer?.cancel();
      _statusTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
        _updateStatus();
      });
    } catch (e) {
      setState(() {
        _statusText = 'Failed to start recording: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    if (!_isRecording) return;

    _statusTimer?.cancel();

    try {
      final path = await stopRecording();
      setState(() {
        _isRecording = false;
        _lastRecordedFile = path;
        _statusText = 'Recording saved: ${path.split('/').last}';
      });

      // Restart slower metrics timer
      _statusTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _updateStatus();
      });
    } catch (e) {
      setState(() {
        _isRecording = false;
        _statusText = 'Failed to stop recording: $e';
      });

      // Restart metrics timer even on error
      _statusTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
        _updateStatus();
      });
    }
  }

  Future<void> _startPlayback() async {
    if (_lastRecordedFile == null || _isPlaying || _isRecording) return;

    try {
      await startPlayback(filePath: _lastRecordedFile!);

      setState(() {
        _isPlaying = true;
        _statusText = 'Playing...';
      });

      // Monitor playback status
      _statusTimer = Timer.periodic(const Duration(milliseconds: 100), (_) async {
        final finished = await isPlaybackFinished();
        if (finished) {
          _statusTimer?.cancel();
          setState(() {
            _isPlaying = false;
            _statusText = 'Playback finished';
          });
        }
      });
    } catch (e) {
      setState(() {
        _statusText = 'Failed to start playback: $e';
      });
    }
  }

  Future<void> _stopPlayback() async {
    if (!_isPlaying) return;

    _statusTimer?.cancel();

    try {
      await stopPlayback();
      setState(() {
        _isPlaying = false;
        _statusText = 'Playback stopped';
      });
    } catch (e) {
      setState(() {
        _statusText = 'Failed to stop playback: $e';
      });
    }
  }

  void _updateMetrics() async {
    try {
      final metrics = await getSystemMetrics();
      setState(() {
        _cpuUsage = metrics.cpuUsagePercent;
        _gpuUsage = metrics.gpuUsagePercent;
        _nnapiAvailable = metrics.nnapiAvailable;
      });
    } catch (e) {
      // Ignore metrics update errors
    }
  }

  void _updateStatus() async {
    try {
      final status = await getStatus();
      final metrics = await getSystemMetrics();
      setState(() {
        _recordingDuration = status.durationMs.toInt();
        _streamAEnabled = status.streamAEnabled;
        _streamBEnabled = status.streamBEnabled;
        _cpuUsage = metrics.cpuUsagePercent;
        _gpuUsage = metrics.gpuUsagePercent;
        _nnapiAvailable = metrics.nnapiAvailable;
      });
    } catch (e) {
      // Ignore status update errors
    }
  }

  Future<void> _toggleStreamA(bool enabled) async {
    try {
      await setStreamAEnabled(enabled: enabled);
      setState(() {
        _streamAEnabled = enabled;
      });
    } catch (e) {
      setState(() {
        _statusText = 'Failed to toggle stream A: $e';
      });
    }
  }

  Future<void> _toggleStreamB(bool enabled) async {
    try {
      await setStreamBEnabled(enabled: enabled);
      setState(() {
        _streamBEnabled = enabled;
      });
    } catch (e) {
      setState(() {
        _statusText = 'Failed to toggle stream B: $e';
      });
    }
  }

  String _formatDuration(int ms) {
    final seconds = (ms / 1000).floor();
    final minutes = (seconds / 60).floor();
    final secs = seconds % 60;
    final millis = (ms % 1000) ~/ 100;
    return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}.$millis';
  }

  Widget _buildMetricColumn(String label, String value, double progress, Color color) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 4),
        SizedBox(
          width: 60,
          height: 60,
          child: Stack(
            alignment: Alignment.center,
            children: [
              CircularProgressIndicator(
                value: progress.clamp(0.0, 1.0),
                strokeWidth: 6,
                backgroundColor: Colors.grey.shade300,
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
              Text(value, style: const TextStyle(fontSize: 12)),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('DeepFilter Test'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Status indicator
              Container(
                width: 200,
                height: 200,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isRecording
                      ? Colors.red.withOpacity(0.2)
                      : _isPlaying
                          ? Colors.green.withOpacity(0.2)
                          : Colors.grey.withOpacity(0.1),
                  border: Border.all(
                    color: _isRecording
                        ? Colors.red
                        : _isPlaying
                            ? Colors.green
                            : Colors.grey,
                    width: 4,
                  ),
                ),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        _isRecording
                            ? Icons.mic
                            : _isPlaying
                                ? Icons.volume_up
                                : Icons.mic_none,
                        size: 64,
                        color: _isRecording
                            ? Colors.red
                            : _isPlaying
                                ? Colors.green
                                : Colors.grey,
                      ),
                      if (_isRecording) ...[
                        const SizedBox(height: 8),
                        Text(
                          _formatDuration(_recordingDuration),
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 32),

              // Status text
              Text(
                _statusText,
                style: Theme.of(context).textTheme.bodyLarge,
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 48),

              // Record button
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: !_isInitialized || _isPlaying
                        ? null
                        : _isRecording
                            ? _stopRecording
                            : _startRecording,
                    icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
                    label: Text(_isRecording ? 'Stop' : 'Record'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isRecording ? Colors.red : null,
                      foregroundColor: _isRecording ? Colors.white : null,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 16,
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  ElevatedButton.icon(
                    onPressed: _lastRecordedFile == null || _isRecording
                        ? null
                        : _isPlaying
                            ? _stopPlayback
                            : _startPlayback,
                    icon: Icon(_isPlaying ? Icons.stop : Icons.play_arrow),
                    label: Text(_isPlaying ? 'Stop' : 'Play'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isPlaying ? Colors.green : null,
                      foregroundColor: _isPlaying ? Colors.white : null,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 32,
                        vertical: 16,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              // Info text (moved here, below buttons)
              Text(
                'Audio is processed with DeepFilterNet3\nfor real-time noise suppression',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 32),

              // Stream controls
              if (_isInitialized) ...[
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Text(
                          'Dual Stream Controls',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Text('Stream A'),
                            Switch(
                              value: _streamAEnabled,
                              onChanged: _toggleStreamA,
                            ),
                            const SizedBox(width: 24),
                            const Text('Stream B'),
                            Switch(
                              value: _streamBEnabled,
                              onChanged: _toggleStreamB,
                            ),
                          ],
                        ),
                        Text(
                          'Both streams process the same input.\nFinal output is the average of enabled streams.',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.grey,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // System Metrics
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Text(
                          'System Metrics',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            _buildMetricColumn(
                              'CPU',
                              _cpuUsage >= 0 ? '${_cpuUsage.toStringAsFixed(1)}%' : 'N/A',
                              _cpuUsage >= 0 ? _cpuUsage / 100 : 0,
                              Colors.blue,
                            ),
                            _buildMetricColumn(
                              'GPU',
                              _gpuUsage >= 0 ? '${_gpuUsage.toStringAsFixed(1)}%' : 'N/A',
                              _gpuUsage >= 0 ? _gpuUsage / 100 : 0,
                              Colors.green,
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              _nnapiAvailable ? Icons.check_circle : Icons.cancel,
                              color: _nnapiAvailable ? Colors.green : Colors.red,
                              size: 18,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'NNAPI: ${_nnapiAvailable ? "Available" : "Not Available"}',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // Bottom padding for navigation bar
              const SizedBox(height: 100),
            ],
          ),
        ),
      ),
    );
  }
}
