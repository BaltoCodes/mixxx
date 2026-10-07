#pragma once

#include <QObject>
#include <QString>
#include <memory>
#include <vector>

#include "analyzer/plugins/analyzerplugin.h"
#include "preferences/usersettings.h"

namespace mixxx {

namespace beatthis {
class Resampler;
class MelSpectrogram;
class Chunker;
} // namespace beatthis

/// Beat tracker based on the "Beat This!" neural network
/// (CP JKU Lines, ISMIR 2024), run with ONNX Runtime.
class AnalyzerBeatThis : public AnalyzerBeatsPlugin {
  public:
    static AnalyzerPluginInfo pluginInfo() {
        return AnalyzerPluginInfo(
                // The model name is part of the ID: a new model changes the
                // ID, which allows re-analysis via AnalyzerBeats::shouldAnalyze().
                // Don't change it without changing the model.
                QStringLiteral("beatthis:small0"),
                QObject::tr("CP JKU Lines (Foscarin, Schlüter, Widmer)"),
                QObject::tr("Beat This! (neural network)"),
                true);
    }

    static QString defaultModelPath(const UserSettingsPointer& pConfig);

    explicit AnalyzerBeatThis(QString modelPath);
    ~AnalyzerBeatThis() override;

    AnalyzerPluginInfo info() const override {
        return pluginInfo();
    }

    bool initialize(mixxx::audio::SampleRate sampleRate) override;
    bool processSamples(const CSAMPLE* pIn, SINT iLen) override;
    bool finalize() override;

    bool supportsBeatTracking() const override {
        return true;
    }

    QVector<mixxx::audio::FramePos> getBeats() const override {
        return m_resultBeats;
    }

    /// Not used by Mixxx yet.
    QVector<mixxx::audio::FramePos> getDownbeats() const {
        return m_resultDownbeats;
    }

  private:
    bool feedSpectrogram();
    void reset();

    const QString m_modelPath;
    mixxx::audio::SampleRate m_sampleRate;

    std::unique_ptr<beatthis::Resampler> m_pResampler;
    std::unique_ptr<beatthis::MelSpectrogram> m_pMel;
    std::unique_ptr<beatthis::Chunker> m_pChunker;
    std::vector<float> m_resampled;

    QVector<mixxx::audio::FramePos> m_resultBeats;
    QVector<mixxx::audio::FramePos> m_resultDownbeats;
};

} // namespace mixxx
