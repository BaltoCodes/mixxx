#pragma once

#include <soxr.h>

#include <vector>

#include "util/types.h"

namespace mixxx::beatthis {

/// Converts the interleaved stereo signal of Mixxx into mono at 22050 Hz
/// using soxr in HQ quality, like the Python reference (soxr.resample).
class Resampler {
  public:
    explicit Resampler(double inputSampleRate);
    ~Resampler();
    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;

    bool isValid() const {
        return m_passthrough || m_pSoxr != nullptr;
    }

    /// Appends the produced samples to *pOut.
    bool processStereo(const CSAMPLE* pIn, SINT numFrames, std::vector<float>* pOut);

    /// Flushes the internal filter at the end of the track.
    bool flush(std::vector<float>* pOut);

  private:
    bool drain(const float* pIn, size_t inLen, std::vector<float>* pOut);

    // The reference skips resampling if the input is already at 22050 Hz.
    const bool m_passthrough;
    const double m_ratio;
    soxr_t m_pSoxr = nullptr;
    std::vector<float> m_mono;
    std::vector<float> m_scratch;
};

} // namespace mixxx::beatthis
