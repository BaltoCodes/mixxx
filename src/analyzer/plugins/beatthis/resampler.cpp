#include "analyzer/plugins/beatthis/resampler.h"

#include <QtDebug>
#include <cmath>

#include "analyzer/plugins/beatthis/constants.h"

namespace mixxx::beatthis {

Resampler::Resampler(double inputSampleRate)
        : m_passthrough(inputSampleRate == kModelSampleRate),
          m_ratio(kModelSampleRate / inputSampleRate) {
    if (m_passthrough) {
        return;
    }
    soxr_error_t error = nullptr;
    const soxr_io_spec_t ioSpec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    const soxr_quality_spec_t qualitySpec = soxr_quality_spec(SOXR_HQ, 0);
    m_pSoxr = soxr_create(inputSampleRate,
            kModelSampleRate,
            1,
            &error,
            &ioSpec,
            &qualitySpec,
            nullptr);
    if (error) {
        qWarning() << "BeatThis: soxr_create failed:" << error;
        soxr_delete(m_pSoxr);
        m_pSoxr = nullptr;
    }
}

Resampler::~Resampler() {
    if (m_pSoxr) {
        soxr_delete(m_pSoxr);
    }
}

bool Resampler::processStereo(const CSAMPLE* pIn, SINT numFrames, std::vector<float>* pOut) {
    if (!isValid()) {
        return false;
    }
    if (numFrames <= 0) {
        return true;
    }
    // Mono = mean of the channels, like signal.mean(1) in the reference.
    m_mono.resize(static_cast<size_t>(numFrames));
    for (SINT i = 0; i < numFrames; ++i) {
        m_mono[i] = 0.5f * (pIn[2 * i] + pIn[2 * i + 1]);
    }
    if (m_passthrough) {
        pOut->insert(pOut->end(), m_mono.begin(), m_mono.end());
        return true;
    }
    return drain(m_mono.data(), m_mono.size(), pOut);
}

bool Resampler::flush(std::vector<float>* pOut) {
    if (!isValid()) {
        return false;
    }
    if (m_passthrough) {
        return true;
    }
    return drain(nullptr, 0, pOut); // in == nullptr signals the end of input to soxr
}

bool Resampler::drain(const float* pIn, size_t inLen, std::vector<float>* pOut) {
    m_scratch.resize(static_cast<size_t>(std::ceil(inLen * m_ratio)) + 1024);
    size_t consumed = 0;
    while (true) {
        size_t inDone = 0;
        size_t outDone = 0;
        const soxr_error_t error = soxr_process(m_pSoxr,
                pIn ? pIn + consumed : nullptr,
                pIn ? inLen - consumed : 0,
                &inDone,
                m_scratch.data(),
                m_scratch.size(),
                &outDone);
        if (error) {
            qWarning() << "BeatThis: soxr_process failed:" << error;
            return false;
        }
        pOut->insert(pOut->end(), m_scratch.begin(), m_scratch.begin() + outDone);
        consumed += inDone;
        const bool inputLeft = pIn && consumed < inLen;
        if (!inputLeft && outDone == 0) {
            return true;
        }
        if (pIn && !inputLeft && outDone < m_scratch.size()) {
            return true; // everything consumed, soxr keeps the rest for later
        }
    }
}

} // namespace mixxx::beatthis
