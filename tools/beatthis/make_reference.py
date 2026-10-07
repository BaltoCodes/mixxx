#!/usr/bin/env python3
"""Generate the reference data for src/test/beatthis_test.cpp.

Usage:
    pip install beat_this soundfile soxr
    python make_reference.py src/test/beatthis/excerpt.wav

The input must be a 16-bit PCM WAV file (the C++ test reads it directly).
All arrays are written as raw little-endian float32 so that the C++ test
can read them without any extra dependency.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio
from beat_this.inference import load_model, split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect


def write_f32(path, array):
    np.ascontiguousarray(array, dtype="<f4").tofile(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--outdir", default="src/test/beatthis")
    p.add_argument("--model", default="small0")
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    info = sf.info(a.audio)
    assert info.subtype == "PCM_16", "the C++ test only reads 16-bit PCM WAV"
    (out / "samplerate.txt").write_text(f"{info.samplerate}\n")

    # Same steps as beat_this.inference.Audio2Frames.signal2spect().
    signal, sr = sf.read(a.audio, dtype="float64")
    if signal.ndim == 2:
        signal = signal.mean(1)
    if sr != 22050:
        signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
    mono22 = signal.astype(np.float32)
    write_f32(out / "mono22050.f32", mono22)

    fb = torchaudio.functional.melscale_fbanks(
        513, 30.0, 11000.0, 128, 22050, norm=None, mel_scale="slaney"
    )
    write_f32(out / "filterbank.f32", fb.numpy())

    spect = LogMelSpect()(torch.tensor(mono22, dtype=torch.float32))
    write_f32(out / "spect.f32", spect.numpy())

    model = load_model(a.model, device="cpu").eval()
    with torch.inference_mode():
        pred = split_predict_aggregate(
            spect,
            chunk_size=1500,
            border_size=6,
            overlap_mode="keep_first",
            model=model,
        )
    write_f32(out / "beat_logits.f32", pred["beat"].numpy())
    write_f32(out / "downbeat_logits.f32", pred["downbeat"].numpy())

    beats, downbeats = Postprocessor(type="minimal")(
        pred["beat"], pred["downbeat"]
    )
    np.savetxt(out / "beats.txt", beats, fmt="%.6f")
    np.savetxt(out / "downbeats.txt", downbeats, fmt="%.6f")
    print(
        f"{spect.shape[0]} frames, {len(beats)} beats, "
        f"{len(downbeats)} downbeats"
    )
    print("Reference written to", out)


if __name__ == "__main__":
    main()
