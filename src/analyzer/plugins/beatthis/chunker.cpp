#include "analyzer/plugins/beatthis/chunker.h"

#include <algorithm>
#include <cmath>

#include "analyzer/plugins/beatthis/constants.h"
#include "analyzer/plugins/beatthis/model.h"

namespace mixxx::beatthis {

// The chunk starts of the reference are np.arange(-6, F - 6, 1488), and the
// last one is moved to F - 1494 if F > 1488 ("avoid_short_end"). A regular
// chunk can be run as soon as start + 1500 <= received frames: the last
// start of the list never fulfills this condition, so no chunk that was run
// early has to be replaced later.

Chunker::Chunker(std::shared_ptr<Model> pModel)
        : m_pModel(std::move(pModel)),
          m_nextChunkStart(-kBorderFrames),
          m_chunkInput(static_cast<size_t>(kChunkFrames) * kNumMels),
          m_chunkBeat(kChunkFrames),
          m_chunkDownbeat(kChunkFrames) {
}

void Chunker::ensureLogitsSize(int64_t size) {
    if (static_cast<int64_t>(m_beatLogits.size()) < size) {
        m_beatLogits.resize(size, kMaskedLogit);
        m_downbeatLogits.resize(size, kMaskedLogit);
        m_written.resize(size, false);
    }
}

bool Chunker::addFrame(const float* pMelFrame) {
    m_spec.insert(m_spec.end(), pMelFrame, pMelFrame + kNumMels);
    ++m_numFrames;

    // Regular chunks: fully available.
    while (m_nextChunkStart + kChunkFrames <= m_numFrames) {
        if (!runChunk(m_nextChunkStart, -1)) {
            return false;
        }
        m_nextChunkStart += kChunkStep;
    }

    // Keep a full chunk before the next start: this covers the shifted
    // last chunk of finish().
    const int64_t keepFrom = std::max<int64_t>(0, m_nextChunkStart - kChunkFrames);
    const int64_t drop = keepFrom - m_specStart;
    if (drop >= kChunkFrames) {
        m_spec.erase(m_spec.begin(), m_spec.begin() + drop * kNumMels);
        m_specStart = keepFrom;
    }
    return true;
}

bool Chunker::runChunk(int64_t start, int64_t totalFrames) {
    const int64_t available = totalFrames < 0 ? m_numFrames : totalFrames;
    const int64_t first = std::max<int64_t>(start, 0);
    const int64_t last = std::min<int64_t>(start + kChunkFrames, available);

    // Zeros everywhere else: left padding, right padding, and filling up
    // to kChunkFrames for pieces shorter than one chunk (the ONNX model
    // has a fixed input length).
    std::fill(m_chunkInput.begin(), m_chunkInput.end(), 0.0f);
    for (int64_t f = first; f < last; ++f) {
        const float* pSrc = &m_spec[static_cast<size_t>(f - m_specStart) * kNumMels];
        std::copy(pSrc,
                pSrc + kNumMels,
                &m_chunkInput[static_cast<size_t>(f - start) * kNumMels]);
    }

    if (!m_pModel->run(m_chunkInput.data(), m_chunkBeat.data(), m_chunkDownbeat.data())) {
        return false;
    }

    // Discard the borders, and never overwrite a value that was already
    // written (keep_first: the chunks are processed in list order).
    const int64_t writeFrom = start + kBorderFrames;
    const int64_t writeTo = std::min<int64_t>(start + kChunkFrames - kBorderFrames, available);
    ensureLogitsSize(writeTo);
    for (int64_t p = std::max<int64_t>(writeFrom, 0); p < writeTo; ++p) {
        if (!m_written[p]) {
            m_beatLogits[p] = m_chunkBeat[p - start];
            m_downbeatLogits[p] = m_chunkDownbeat[p - start];
            m_written[p] = true;
        }
    }
    return true;
}

bool Chunker::finish() {
    const int64_t total = m_numFrames;
    if (total <= 0) {
        return false;
    }
    ensureLogitsSize(total);

    // Remaining starts of np.arange(-6, F - 6, 1488).
    std::vector<int64_t> starts;
    for (int64_t s = m_nextChunkStart; s < total - kBorderFrames; s += kChunkStep) {
        starts.push_back(s);
    }
    // avoid_short_end: shift the last start of the complete list.
    if (!starts.empty() && total > kChunkStep) {
        starts.back() = total - (kChunkFrames - kBorderFrames);
    }
    for (const int64_t s : starts) {
        if (!runChunk(s, total)) {
            return false;
        }
    }
    m_beatLogits.resize(total);
    m_downbeatLogits.resize(total);
    m_spec.clear();
    m_spec.shrink_to_fit();
    return true;
}

// static
std::vector<double> Chunker::pickPeaks(const std::vector<float>& logits) {
    const int64_t n = static_cast<int64_t>(logits.size());
    const int half = kPeakPoolSize / 2;

    // 1. Local maxima (max_pool1d(7, 1, 3), -inf padding) with logit > 0.
    std::vector<int64_t> peaks;
    for (int64_t i = 0; i < n; ++i) {
        const float v = logits[i];
        if (v <= 0.0f) {
            continue;
        }
        // Starts from v instead of -inf: Mixxx is built with -ffast-math.
        float localMax = v;
        for (int64_t j = std::max<int64_t>(0, i - half);
                j <= std::min<int64_t>(n - 1, i + half);
                ++j) {
            localMax = std::max(localMax, logits[j]);
        }
        if (v == localMax) {
            peaks.push_back(i);
        }
    }

    // 2. deduplicate_peaks(width=1): running mean of adjacent peaks,
    //    reproduced exactly (including its way of computing the mean).
    std::vector<double> frames;
    if (!peaks.empty()) {
        double p = static_cast<double>(peaks[0]);
        int count = 1;
        for (size_t k = 1; k < peaks.size(); ++k) {
            const double p2 = static_cast<double>(peaks[k]);
            if (p2 - p <= 1.0) {
                ++count;
                p += (p2 - p) / count;
            } else {
                frames.push_back(p);
                p = p2;
                count = 1;
            }
        }
        frames.push_back(p);
    }

    // 3. Frames -> seconds.
    for (double& f : frames) {
        f /= kFramesPerSecond;
    }
    return frames;
}

std::vector<double> Chunker::beatTimes() const {
    return pickPeaks(m_beatLogits);
}

std::vector<double> Chunker::downbeatTimes() const {
    const std::vector<double> beats = beatTimes();
    std::vector<double> downbeats = pickPeaks(m_downbeatLogits);
    if (beats.empty()) {
        return downbeats;
    }
    // Move every downbeat to the nearest beat (np.argmin returns the
    // first minimum, like std::min_element with a strict comparison),
    // then remove duplicates (np.unique also sorts).
    for (double& d : downbeats) {
        const auto it = std::min_element(beats.begin(), beats.end(), [d](double a, double b) {
            return std::abs(a - d) < std::abs(b - d);
        });
        d = *it;
    }
    std::sort(downbeats.begin(), downbeats.end());
    downbeats.erase(std::unique(downbeats.begin(), downbeats.end()), downbeats.end());
    return downbeats;
}

} // namespace mixxx::beatthis
