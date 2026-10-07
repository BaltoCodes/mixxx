#pragma once

#include <cstdint>

namespace mixxx::beatthis {

// All values are taken from beat_this/preprocessing.py and
// beat_this/inference.py. They must never be changed without
// re-exporting (and re-training) the model.

// Preprocessing (LogMelSpect)
constexpr double kModelSampleRate = 22050.0;
constexpr int kFftSize = 1024;
constexpr int kHopSize = 441;               // 22050 / 441 = 50 frames/s
constexpr int kNumFreqs = kFftSize / 2 + 1; // 513
constexpr int kNumMels = 128;
constexpr double kMelFMin = 30.0;
constexpr double kMelFMax = 11000.0;
constexpr double kLogMultiplier = 1000.0;
constexpr double kFramesPerSecond = kModelSampleRate / kHopSize; // 50

// Inference (split_predict_aggregate)
constexpr int kChunkFrames = 1500; // 30 s, fixed input length of the ONNX model
constexpr int kBorderFrames = 6;
constexpr int kChunkStep = kChunkFrames - 2 * kBorderFrames; // 1488

// Postprocessing (Postprocessor.postp_minimal)
constexpr int kPeakPoolSize = 7; // +/- 70 ms
constexpr float kMaskedLogit = -1000.0f;

// The "reflect" padding of torch.stft(center=True) needs more than
// kFftSize / 2 input samples.
constexpr int64_t kMinSamples = kFftSize / 2 + 1;

} // namespace mixxx::beatthis
