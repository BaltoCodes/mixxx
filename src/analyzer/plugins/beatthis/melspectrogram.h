#pragma once

#include <fftw3.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace mixxx::beatthis {

/// Log-mel spectrogram identical to beat_this.preprocessing.LogMelSpect,
/// computed on the fly to bound the memory usage.
class MelSpectrogram {
  public:
    /// Receives one frame of kNumMels values.
    using FrameCallback = std::function<bool(const float* pMelFrame)>;

    MelSpectrogram();
    ~MelSpectrogram();
    MelSpectrogram(const MelSpectrogram&) = delete;
    MelSpectrogram& operator=(const MelSpectrogram&) = delete;

    /// Mono samples at 22050 Hz.
    bool process(const float* pSamples, size_t count, const FrameCallback& callback);

    /// Computes the last frames (with the "reflect" padding at the end).
    /// Returns false if the signal is too short for the padding.
    bool finish(const FrameCallback& callback);

    int64_t receivedSamples() const {
        return m_received;
    }

    /// Exposed for tests. Layout: [kNumFreqs][kNumMels], row = frequency.
    static std::vector<float> makeSlaneyFilterbank();

  private:
    /// Sample at the absolute position `index`, reflected at the edges.
    /// `totalLength` < 0: the length is still unknown (no reflection at the end).
    float sampleAt(int64_t index, int64_t totalLength) const;
    bool computeFrame(int64_t frameIndex, int64_t totalLength, const FrameCallback& callback);
    void discardOldSamples();

    std::vector<double> m_window;
    std::vector<float> m_filterbank; // [kNumFreqs][kNumMels], row = frequency
    std::vector<float> m_buffer;     // samples starting at m_bufferStart
    int64_t m_bufferStart = 0;
    int64_t m_received = 0;
    int64_t m_nextFrame = 0;

    double* m_pFftIn = nullptr;
    fftw_complex* m_pFftOut = nullptr;
    fftw_plan m_plan = nullptr;
    std::vector<float> m_magnitude;
    std::vector<float> m_melFrame;
};

} // namespace mixxx::beatthis
