# Neural network models

## `beatthis_small0.onnx`

Beat and downbeat tracker used by the "Beat This! (neural network)" beat
detection plugin (CMake option `BEATTHIS`).

| | |
| --- | --- |
| Source | <https://github.com/CPJKU/beat_this>, checkpoint `small0` |
| `beat_this` version | 1.1.0 |
| Export | `tools/beatthis/export_onnx.py --model small0` (PyTorch 2.7.0, opset 17) |
| Input | `spect`: float32 `(1, 1500, 128)` log-mel spectrogram, 50 frames/s |
| Outputs | `beat`, `downbeat`: float32 `(1, 1500)` logits |
| SHA-256 | `ae244a7b8c2448ed7618c9ee1992ef34b11c93a8e410db8ac0ce43e654cd0432` |
| License | MIT (code and weights) |

Citation:

> F. Foscarin, J. Schlüter, G. Widmer, *Beat This! Accurate Beat Tracking
> Without DBN Postprocessing*, Proceedings of the 25th International Society
> for Music Information Retrieval Conference (ISMIR), 2024.
> <https://arxiv.org/abs/2407.21658>

The authors note that part of the training data is under restrictive
licenses; the weights themselves are published under the MIT license.

The preprocessing constants in `src/analyzer/plugins/beatthis/constants.h`
must match the model. Changing the model requires changing the plugin ID in
`AnalyzerBeatThis::pluginInfo()`, so that tracks get re-analyzed.
