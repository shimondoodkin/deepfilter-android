use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use anyhow::{Result, Context};
use hound::{WavSpec, WavWriter as HoundWriter, WavReader as HoundReader, SampleFormat};

/// WAV file writer for 48kHz mono float audio
pub struct WavWriter {
    writer: HoundWriter<BufWriter<File>>,
}

impl WavWriter {
    pub fn new(path: &str, sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let file = File::create(path)
            .context(format!("Failed to create file: {}", path))?;
        let writer = HoundWriter::new(BufWriter::new(file), spec)
            .context("Failed to create WAV writer")?;

        Ok(Self { writer })
    }

    /// Write f32 samples, converting to i16
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<()> {
        for &sample in samples {
            // Clamp and convert to i16
            let clamped = sample.clamp(-1.0, 1.0);
            let i16_sample = (clamped * i16::MAX as f32) as i16;
            self.writer.write_sample(i16_sample)
                .context("Failed to write sample")?;
        }
        Ok(())
    }

    /// Finalize the WAV file
    pub fn finalize(self) -> Result<()> {
        self.writer.finalize()
            .context("Failed to finalize WAV file")?;
        Ok(())
    }
}

/// WAV file reader
pub struct WavReader {
    reader: HoundReader<std::io::BufReader<File>>,
    sample_rate: u32,
    channels: u16,
}

impl WavReader {
    pub fn new(path: &str) -> Result<Self> {
        let reader = HoundReader::open(path)
            .context(format!("Failed to open WAV file: {}", path))?;

        let spec = reader.spec();

        Ok(Self {
            reader,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Read all samples as f32
    pub fn read_all_samples(self) -> Result<Vec<f32>> {
        let spec = self.reader.spec();

        let samples: Result<Vec<f32>, _> = match spec.sample_format {
            SampleFormat::Int => {
                match spec.bits_per_sample {
                    16 => self.reader
                        .into_samples::<i16>()
                        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                        .collect(),
                    32 => self.reader
                        .into_samples::<i32>()
                        .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                        .collect(),
                    8 => self.reader
                        .into_samples::<i8>()
                        .map(|s| s.map(|v| v as f32 / i8::MAX as f32))
                        .collect(),
                    _ => anyhow::bail!("Unsupported bit depth: {}", spec.bits_per_sample),
                }
            }
            SampleFormat::Float => {
                self.reader
                    .into_samples::<f32>()
                    .collect()
            }
        };

        samples.context("Failed to read WAV samples")
    }
}
