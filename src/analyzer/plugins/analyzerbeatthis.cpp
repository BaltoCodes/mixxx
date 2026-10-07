#include "analyzer/plugins/analyzerbeatthis.h"

#include <QDir>
#include <QtDebug>

#include "analyzer/constants.h"
#include "analyzer/plugins/beatthis/chunker.h"
#include "analyzer/plugins/beatthis/melspectrogram.h"
#include "analyzer/plugins/beatthis/model.h"
#include "analyzer/plugins/beatthis/resampler.h"
#include "util/assert.h"

namespace mixxx {

// static
QString AnalyzerBeatThis::defaultModelPath(const UserSettingsPointer& pConfig) {
    return QDir(pConfig->getResourcePath())
            .filePath(QStringLiteral("models/beatthis_small0.onnx"));
}

AnalyzerBeatThis::AnalyzerBeatThis(QString modelPath)
        : m_modelPath(std::move(modelPath)) {
}

AnalyzerBeatThis::~AnalyzerBeatThis() = default;

void AnalyzerBeatThis::reset() {
    m_pChunker.reset();
    m_pMel.reset();
    m_pResampler.reset();
    m_resampled.clear();
    m_resampled.shrink_to_fit();
}

bool AnalyzerBeatThis::initialize(mixxx::audio::SampleRate sampleRate) {
    reset();
    m_resultBeats.clear();
    m_resultDownbeats.clear();
    m_sampleRate = sampleRate;

    auto pModel = beatthis::Model::load(m_modelPath);
    if (!pModel) {
        qWarning() << "BeatThis: model unavailable, analysis disabled:" << m_modelPath;
        return false;
    }
    m_pResampler = std::make_unique<beatthis::Resampler>(sampleRate.toDouble());
    if (!m_pResampler->isValid()) {
        reset();
        return false;
    }
    m_pMel = std::make_unique<beatthis::MelSpectrogram>();
    m_pChunker = std::make_unique<beatthis::Chunker>(std::move(pModel));
    return true;
}

bool AnalyzerBeatThis::feedSpectrogram() {
    return m_pMel->process(m_resampled.data(),
            m_resampled.size(),
            [this](const float* pFrame) { return m_pChunker->addFrame(pFrame); });
}

bool AnalyzerBeatThis::processSamples(const CSAMPLE* pIn, SINT iLen) {
    DEBUG_ASSERT(iLen % kAnalysisChannels == 0);
    if (!m_pResampler) {
        return false;
    }
    m_resampled.clear();
    if (!m_pResampler->processStereo(pIn, iLen / kAnalysisChannels, &m_resampled)) {
        return false;
    }
    return feedSpectrogram();
}

bool AnalyzerBeatThis::finalize() {
    if (!m_pResampler) {
        return false;
    }
    m_resampled.clear();
    const auto addFrame = [this](const float* pFrame) {
        return m_pChunker->addFrame(pFrame);
    };
    const bool ok = m_pResampler->flush(&m_resampled) &&
            feedSpectrogram() &&
            m_pMel->finish(addFrame) &&
            m_pChunker->finish();
    if (!ok) {
        qWarning() << "BeatThis: analysis failed";
        reset();
        return false;
    }

    const double sampleRate = m_sampleRate.toDouble();
    const auto toFramePos = [sampleRate](double seconds) {
        return mixxx::audio::FramePos(seconds * sampleRate);
    };
    for (const double t : m_pChunker->beatTimes()) {
        m_resultBeats.push_back(toFramePos(t));
    }
    for (const double t : m_pChunker->downbeatTimes()) {
        m_resultDownbeats.push_back(toFramePos(t));
    }
    qDebug() << "BeatThis: detected" << m_resultBeats.size() << "beats and"
             << m_resultDownbeats.size() << "downbeats";

    reset();
    return true;
}

} // namespace mixxx
