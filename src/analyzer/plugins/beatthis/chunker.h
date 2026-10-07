#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace mixxx::beatthis {

class Model;

/// Streaming equivalent of beat_this.inference.split_predict_aggregate()
/// (overlap_mode="keep_first") followed by Postprocessor.postp_minimal().
/// Only about two chunks of the spectrogram are kept in memory.
class Chunker {
  public:
    explicit Chunker(std::shared_ptr<Model> pModel);

    /// Appends one frame of kNumMels values.
    bool addFrame(const float* pMelFrame);

    /// Runs the remaining chunks. Must be called exactly once.
    bool finish();

    /// Times in seconds ("minimal" postprocessing). Valid after finish().
    std::vector<double> beatTimes() const;
    std::vector<double> downbeatTimes() const;

    /// Exposed for tests.
    const std::vector<float>& beatLogits() const {
        return m_beatLogits;
    }
    const std::vector<float>& downbeatLogits() const {
        return m_downbeatLogits;
    }
    static std::vector<double> pickPeaks(const std::vector<float>& logits);

  private:
    /// totalFrames < 0: regular chunk, all of its frames have been received.
    bool runChunk(int64_t start, int64_t totalFrames);
    void ensureLogitsSize(int64_t size);

    std::shared_ptr<Model> m_pModel;

    // Frames kept from m_specStart on (row = frame).
    std::vector<float> m_spec;
    int64_t m_specStart = 0;
    int64_t m_numFrames = 0;

    int64_t m_nextChunkStart; // starts at -kBorderFrames

    std::vector<float> m_beatLogits;
    std::vector<float> m_downbeatLogits;
    std::vector<bool> m_written;

    // Reused buffers, to avoid allocations per chunk.
    std::vector<float> m_chunkInput;
    std::vector<float> m_chunkBeat;
    std::vector<float> m_chunkDownbeat;
};

} // namespace mixxx::beatthis
