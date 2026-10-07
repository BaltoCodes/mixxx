#!/usr/bin/env python3
"""Export a Beat This! checkpoint to ONNX for Mixxx.

Usage:
    pip install beat_this onnx onnxruntime
    python export_onnx.py --model small0 --out res/models/beatthis_small0.onnx

The model is exported with a fixed input length of CHUNK_FRAMES frames,
which is the chunk length used by split_predict_aggregate(). This avoids
export issues with the rotary embeddings on dynamic lengths.
"""

import argparse
import hashlib

import numpy as np
import onnxruntime as ort
import torch
from beat_this.inference import load_model

CHUNK_FRAMES = 1500
N_MELS = 128


class Wrapper(torch.nn.Module):
    """The model returns a dict, ONNX wants a tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spect):
        out = self.model(spect)
        return out["beat"], out["downbeat"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = load_model(args.model, device="cpu").eval()
    wrapper = Wrapper(model).eval()

    dummy = torch.randn(1, CHUNK_FRAMES, N_MELS)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            args.out,
            input_names=["spect"],
            output_names=["beat", "downbeat"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # ONNX Runtime must reproduce the PyTorch outputs.
    session = ort.InferenceSession(
        args.out, providers=["CPUExecutionProvider"]
    )
    for _ in range(5):
        x = torch.rand(1, CHUNK_FRAMES, N_MELS) * 8.0  # realistic magnitude
        with torch.no_grad():
            ref_beat, ref_down = wrapper(x)
        beat, down = session.run(None, {"spect": x.numpy()})
        np.testing.assert_allclose(
            beat, ref_beat.numpy(), atol=1e-3, rtol=1e-3
        )
        np.testing.assert_allclose(
            down, ref_down.numpy(), atol=1e-3, rtol=1e-3
        )

    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"OK: {args.out}\nsha256: {digest}")


if __name__ == "__main__":
    main()
