#!/usr/bin/env python3
"""Synthesize a royalty-free test excerpt for src/test/beatthis_test.cpp.

The excerpt is 32 s long, which is longer than one model chunk (1500
frames = 30 s), so the tests cover the chunking and the shifted last
chunk ("avoid_short_end"). It starts with 4 s of pads without any
percussion, then a 4/4 groove at 124 BPM with a few syncopations.

Usage:
    python make_test_audio.py --out src/test/beatthis/excerpt.wav
"""

import argparse

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
DURATION = 32.0
BPM = 124.0
INTRO = 4.0


def env(n, decay):
    t = np.arange(n) / SAMPLE_RATE
    return np.exp(-t / decay)


def kick(n):
    t = np.arange(n) / SAMPLE_RATE
    freq = 50.0 + 100.0 * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return np.sin(phase) * env(n, 0.15)


def snare(n, rng):
    t = np.arange(n) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 190.0 * t) * env(n, 0.05)
    noise = rng.uniform(-1, 1, n) * env(n, 0.08)
    return 0.4 * tone + 0.6 * noise


def hihat(n, rng):
    noise = rng.uniform(-1, 1, n)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass
    return 0.5 * noise * env(n, 0.02)


def add(dst, src, at, gain):
    start = int(round(at * SAMPLE_RATE))
    end = min(len(dst), start + len(src))
    if start < len(dst):
        dst[start:end] += gain * src[: end - start]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rng = np.random.default_rng(1234)
    n = int(DURATION * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    left = np.zeros(n)
    right = np.zeros(n)

    # Pads over the whole excerpt (A minor / F major / C major / G major).
    chords = [
        (220.0, 261.63, 329.63),
        (174.61, 220.0, 261.63),
        (261.63, 329.63, 392.0),
        (196.0, 246.94, 293.66),
    ]
    beat = 60.0 / BPM
    bar = 4 * beat
    for i, start in enumerate(np.arange(0.0, DURATION, bar)):
        seg = (t >= start) & (t < start + bar)
        for k, f in enumerate(chords[i % len(chords)]):
            tone = 0.06 * np.sin(2 * np.pi * f * t[seg] + k)
            left[seg] += tone * (1.0 if k != 2 else 0.6)
            right[seg] += tone * (1.0 if k != 0 else 0.6)

    k = kick(int(0.3 * SAMPLE_RATE))
    s = snare(int(0.25 * SAMPLE_RATE), rng)
    beat_idx = 0
    for at in np.arange(INTRO, DURATION, beat):
        add(left, k, at, 0.8)
        add(right, k, at, 0.8)
        if beat_idx % 2 == 1:
            add(left, s, at, 0.35)
            add(right, s, at, 0.45)
        # off-beat hats, and a syncopated 16th every other bar
        h = hihat(int(0.08 * SAMPLE_RATE), rng)
        add(left, h, at + beat / 2, 0.25)
        add(right, h, at + beat / 2, 0.3)
        if beat_idx % 8 == 7:
            add(left, h, at + 3 * beat / 4, 0.2)
            add(right, h, at + 3 * beat / 4, 0.2)
        beat_idx += 1

    # Bass line on the groove part.
    bass_notes = [55.0, 43.65, 65.41, 49.0]
    for i, start in enumerate(np.arange(INTRO, DURATION, bar)):
        for j in (0.0, 1.5, 2.0, 3.5):
            at = start + j * beat
            m = int(0.2 * SAMPLE_RATE)
            tt = np.arange(m) / SAMPLE_RATE
            note = np.sin(2 * np.pi * bass_notes[i % 4] * tt) * env(m, 0.12)
            add(left, note, at, 0.3)
            add(right, note, at, 0.3)

    stereo = np.stack([left, right], axis=1)
    stereo *= 0.9 / np.max(np.abs(stereo))
    sf.write(a.out, stereo, SAMPLE_RATE, subtype="PCM_16")
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
