#include "analyzer/plugins/beatthis/melspectrogram.h"

#include <QMutex>
#include <algorithm>
#include <cmath>

#include "analyzer/plugins/beatthis/constants.h"

namespace mixxx::beatthis {
namespace {

// FFTW plan creation and destruction are not thread-safe, and Mixxx may
// analyze several tracks in parallel.
QMutex s_fftwPlanMutex;

constexpr double kPi = 3.14159265358979323846;

// "slaney" mel scale (torchaudio.functional._hz_to_mel / _mel_to_hz).
constexpr double kFSp = 200.0 / 3.0;
constexpr double kMinLogHz = 1000.0;
constexpr double kMinLogMel = kMinLogHz / kFSp; // 15
const double kLogStep = std::log(6.4) / 27.0;

double hzToMel(double hz) {
    if (hz < kMinLogHz) {
        return hz / kFSp;
    }
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

double melToHz(double mel) {
    if (mel < kMinLogMel) {
        return mel * kFSp;
    }
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
}

} // namespace

// static
std::vector<float> MelSpectrogram::makeSlaneyFilterbank() {
    // Same as torchaudio.functional.melscale_fbanks(norm=None, mel_scale="slaney").
    std::vector<double> allFreqs(kNumFreqs);
    for (int i = 0; i < kNumFreqs; ++i) {
        allFreqs[i] = (kModelSampleRate / 2.0) * i / (kNumFreqs - 1);
    }
    const double melMin = hzToMel(kMelFMin);
    const double melMax = hzToMel(kMelFMax);
    std::vector<double> fPts(kNumMels + 2);
    for (int i = 0; i < kNumMels + 2; ++i) {
        fPts[i] = melToHz(melMin + (melMax - melMin) * i / (kNumMels + 1));
    }
    std::vector<float> fb(static_cast<size_t>(kNumFreqs) * kNumMels, 0.0f);
    for (int f = 0; f < kNumFreqs; ++f) {
        for (int m = 0; m < kNumMels; ++m) {
            const double down = (allFreqs[f] - fPts[m]) / (fPts[m + 1] - fPts[m]);
            const double up = (fPts[m + 2] - allFreqs[f]) / (fPts[m + 2] - fPts[m + 1]);
            fb[static_cast<size_t>(f) * kNumMels + m] =
                    static_cast<float>(std::max(0.0, std::min(down, up)));
        }
    }
    return fb;
}

MelSpectrogram::MelSpectrogram()
        : m_window(kFftSize),
          m_filterbank(makeSlaneyFilterbank()),
          m_magnitude(kNumFreqs),
          m_melFrame(kNumMels) {
    for (int i = 0; i < kFftSize; ++i) {
        // periodic Hann window, like torch.hann_window(periodic=True)
        m_window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / kFftSize);
    }
    QMutexLocker lock(&s_fftwPlanMutex);
    m_pFftIn = fftw_alloc_real(kFftSize);
    m_pFftOut = fftw_alloc_complex(kNumFreqs);
    m_plan = fftw_plan_dft_r2c_1d(kFftSize, m_pFftIn, m_pFftOut, FFTW_ESTIMATE);
}

MelSpectrogram::~MelSpectrogram() {
    QMutexLocker lock(&s_fftwPlanMutex);
    fftw_destroy_plan(m_plan);
    fftw_free(m_pFftIn);
    fftw_free(m_pFftOut);
}

float MelSpectrogram::sampleAt(int64_t index, int64_t totalLength) const {
    if (index < 0) {
        index = -index; // reflect at the start: x[-j] = x[j]
    }
    if (totalLength >= 0 && index >= totalLength) {
        index = 2 * (totalLength - 1) - index; // reflect at the end
    }
    return m_buffer[static_cast<size_t>(index - m_bufferStart)];
}

bool MelSpectrogram::computeFrame(
        int64_t frameIndex, int64_t totalLength, const FrameCallback& callback) {
    const int64_t center = frameIndex * kHopSize;
    const int64_t first = center - kFftSize / 2;
    for (int i = 0; i < kFftSize; ++i) {
        m_pFftIn[i] = m_window[i] * sampleAt(first + i, totalLength);
    }
    fftw_execute(m_plan);

    // normalized="frame_length" is torch.stft(normalized=True),
    // which scales by 1 / sqrt(n_fft).
    const double norm = 1.0 / std::sqrt(static_cast<double>(kFftSize));
    for (int f = 0; f < kNumFreqs; ++f) {
        const double re = m_pFftOut[f][0];
        const double im = m_pFftOut[f][1];
        m_magnitude[f] = static_cast<float>(std::sqrt(re * re + im * im) * norm);
    }

    std::fill(m_melFrame.begin(), m_melFrame.end(), 0.0f);
    for (int f = 0; f < kNumFreqs; ++f) {
        const float mag = m_magnitude[f];
        const float* pRow = &m_filterbank[static_cast<size_t>(f) * kNumMels];
        for (int m = 0; m < kNumMels; ++m) {
            m_melFrame[m] += pRow[m] * mag;
        }
    }
    for (int m = 0; m < kNumMels; ++m) {
        m_melFrame[m] = static_cast<float>(std::log1p(kLogMultiplier * m_melFrame[m]));
    }
    return callback(m_melFrame.data());
}

bool MelSpectrogram::process(const float* pSamples, size_t count, const FrameCallback& callback) {
    m_buffer.insert(m_buffer.end(), pSamples, pSamples + count);
    m_received += static_cast<int64_t>(count);

    // A frame can be computed as soon as its whole window has been received
    // (last index center + kFftSize / 2 - 1) and the reflection at the
    // start is possible.
    if (m_received < kMinSamples) {
        return true;
    }
    while (m_nextFrame * kHopSize + kFftSize / 2 - 1 < m_received) {
        if (!computeFrame(m_nextFrame, -1, callback)) {
            return false;
        }
        ++m_nextFrame;
    }
    discardOldSamples();
    return true;
}

bool MelSpectrogram::finish(const FrameCallback& callback) {
    const int64_t total = m_received;
    if (total < kMinSamples) {
        return false; // too short for the reflect padding
    }
    const int64_t numFrames = 1 + total / kHopSize;
    while (m_nextFrame < numFrames) {
        if (!computeFrame(m_nextFrame, total, callback)) {
            return false;
        }
        ++m_nextFrame;
    }
    return true;
}

void MelSpectrogram::discardOldSamples() {
    // Keep a full window before the center of the next frame. This also
    // covers the reflection at the end.
    const int64_t keepFrom = std::max<int64_t>(0, m_nextFrame * kHopSize - kFftSize);
    const int64_t toDrop = keepFrom - m_bufferStart;
    if (toDrop > 64 * kFftSize) { // compact in large blocks
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + toDrop);
        m_bufferStart = keepFrom;
    }
}

} // namespace mixxx::beatthis
