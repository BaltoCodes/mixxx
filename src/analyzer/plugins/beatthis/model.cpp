#include "analyzer/plugins/beatthis/model.h"

#include <QFileInfo>
#include <QHash>
#include <QMutex>
#include <QtDebug>
#include <algorithm>
#include <array>

#include "analyzer/plugins/beatthis/constants.h"

namespace mixxx::beatthis {
namespace {

// The environment is declared before the cache, so the cached sessions
// are destroyed before the environment at program exit.
struct Runtime {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mixxx-beatthis"};
    QMutex mutex;
    // Strong references: analyzers are created and destroyed for every
    // track, so a weak cache would reload the model for each track.
    QHash<QString, std::shared_ptr<Model>> cache;
};

Runtime& runtime() {
    static Runtime s_runtime;
    return s_runtime;
}

constexpr std::array<const char*, 1> kInputNames = {"spect"};
constexpr std::array<const char*, 2> kOutputNames = {"beat", "downbeat"};

} // namespace

Model::Model(Ort::Session session)
        : m_session(std::move(session)) {
}

// static
std::shared_ptr<Model> Model::load(const QString& modelPath) {
    Runtime& rt = runtime();
    QMutexLocker lock(&rt.mutex);
    if (auto pExisting = rt.cache.value(modelPath)) {
        return pExisting;
    }
    if (!QFileInfo::exists(modelPath)) {
        qWarning() << "BeatThis: model file not found:" << modelPath;
        return nullptr;
    }
    try {
        Ort::SessionOptions options;
        // Mixxx already runs several analyzer threads in parallel:
        // one thread per inference avoids oversubscribing the CPU.
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const std::wstring path = modelPath.toStdWString();
#else
        const std::string path = modelPath.toStdString();
#endif
        Ort::Session session(rt.env, path.c_str(), options);
        auto pModel = std::make_shared<Model>(std::move(session));
        rt.cache.insert(modelPath, pModel);
        return pModel;
    } catch (const Ort::Exception& e) {
        qWarning() << "BeatThis: failed to load model" << modelPath << ":" << e.what();
        return nullptr;
    }
}

bool Model::run(const float* pSpect, float* pBeat, float* pDownbeat) {
    try {
        const auto memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> inputShape = {1, kChunkFrames, kNumMels};
        Ort::Value input = Ort::Value::CreateTensor<float>(memoryInfo,
                const_cast<float*>(pSpect),
                static_cast<size_t>(kChunkFrames) * kNumMels,
                inputShape.data(),
                inputShape.size());

        auto outputs = m_session.Run(Ort::RunOptions{nullptr},
                kInputNames.data(),
                &input,
                kInputNames.size(),
                kOutputNames.data(),
                kOutputNames.size());

        for (const auto& output : outputs) {
            if (output.GetTensorTypeAndShapeInfo().GetElementCount() !=
                    static_cast<size_t>(kChunkFrames)) {
                qWarning() << "BeatThis: unexpected model output size";
                return false;
            }
        }
        const float* pBeatOut = outputs[0].GetTensorData<float>();
        const float* pDownOut = outputs[1].GetTensorData<float>();
        std::copy(pBeatOut, pBeatOut + kChunkFrames, pBeat);
        std::copy(pDownOut, pDownOut + kChunkFrames, pDownbeat);
        return true;
    } catch (const Ort::Exception& e) {
        qWarning() << "BeatThis: inference failed:" << e.what();
        return false;
    }
}

} // namespace mixxx::beatthis
