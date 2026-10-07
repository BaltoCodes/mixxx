#pragma once

#include <onnxruntime_cxx_api.h>

#include <QString>
#include <memory>

namespace mixxx::beatthis {

/// ONNX Runtime session shared by all analyses.
/// Ort::Session::Run() is thread-safe: a single session serves all
/// analyzer threads, and the model is loaded only once per program run.
class Model {
  public:
    /// Returns the cached session for this file, or loads it.
    /// nullptr on failure (missing file, invalid model, ...).
    static std::shared_ptr<Model> load(const QString& modelPath);

    /// pSpect: kChunkFrames x kNumMels values (row = frame).
    /// pBeat, pDownbeat: kChunkFrames logits each.
    bool run(const float* pSpect, float* pBeat, float* pDownbeat);

    explicit Model(Ort::Session session);

  private:
    Ort::Session m_session;
};

} // namespace mixxx::beatthis
