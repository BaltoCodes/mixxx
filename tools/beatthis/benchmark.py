#!/usr/bin/env python3
"""Compare the beat trackers of Mixxx against annotations with mir_eval.

1. Extract the beats with the disabled benchmark test:

    MIXXX_BEATTHIS_BENCHMARK_DIR=/path/to/audio \\
    MIXXX_BEATTHIS_BENCHMARK_OUT=/path/to/output \\
    mixxx-test --gtest_also_run_disabled_tests \\
        --gtest_filter=BeatThisBenchmark.DISABLED_Benchmark

   It writes <plugin>.beats.tsv (relative path, then the beat times in
   seconds) and <plugin>.grid.tsv (relative path, BPM, first beat, duration
   of the constant tempo grid built by BeatFactory) for every plugin.

2. Evaluate:

    pip install mir_eval numpy
    python benchmark.py --results /path/to/output --annotations /path/to/beats

   Annotations are one text file per audio file, at the same relative path
   with the extension replaced by --annotation-ext (default ".beats"). The
   first column of each line is the beat time in seconds (the GTZAN beat
   annotations of Marchand et al. use this format).
"""

import argparse
import math
from pathlib import Path

import mir_eval
import numpy as np


def read_beats_tsv(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        if fields and fields[0]:
            result[fields[0]] = np.array([float(x) for x in fields[1:]])
    return result


def read_grid_tsv(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        if len(fields) == 4:
            bpm, first, duration = (float(x) for x in fields[1:])
            if bpm > 0:
                period = 60.0 / bpm
                # extend the grid backwards from the first beat
                start = first - math.floor(first / period) * period
                result[fields[0]] = (bpm, np.arange(start, duration, period))
            else:
                result[fields[0]] = (0.0, np.array([]))
    return result


def read_annotation(path):
    times = []
    for line in Path(path).read_text().splitlines():
        fields = line.replace(",", " ").split()
        if fields:
            times.append(float(fields[0]))
    return np.array(times)


def octave_error(bpm, ref_beats):
    """Classifies the grid BPM against the median annotated tempo."""
    if bpm <= 0 or len(ref_beats) < 2:
        return "none"
    ref_bpm = 60.0 / np.median(np.diff(ref_beats))
    for name, factor in (
        ("ok", 1.0),
        ("x2", 2.0),
        ("/2", 0.5),
        ("x3/2", 1.5),
        ("x2/3", 2 / 3),
    ):
        if abs(bpm / (ref_bpm * factor) - 1.0) < 0.04:
            return name
    return "other"


def evaluate(estimates, annotations):
    scores = {"F-measure": [], "CMLt": [], "AMLt": []}
    for key, est in estimates.items():
        if key not in annotations:
            continue
        ref = mir_eval.beat.trim_beats(annotations[key])
        est = mir_eval.beat.trim_beats(est)
        scores["F-measure"].append(
            mir_eval.beat.f_measure(ref, est, f_measure_threshold=0.07)
        )
        _, cmlt, _, amlt = mir_eval.beat.continuity(ref, est)
        scores["CMLt"].append(cmlt)
        scores["AMLt"].append(amlt)
    return {
        k: (np.mean(v) if v else float("nan")) for k, v in scores.items()
    }, len(scores["CMLt"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results",
        required=True,
        help="output directory of the benchmark test",
    )
    p.add_argument(
        "--annotations",
        required=True,
        help="root directory of the annotations",
    )
    p.add_argument("--annotation-ext", default=".beats")
    a = p.parse_args()

    results = Path(a.results)
    plugins = sorted(
        f.name[: -len(".beats.tsv")] for f in results.glob("*.beats.tsv")
    )
    if not plugins:
        raise SystemExit(f"no *.beats.tsv in {results}")

    annotations = {}
    for key in read_beats_tsv(results / f"{plugins[0]}.beats.tsv"):
        path = (Path(a.annotations) / key).with_suffix(a.annotation_ext)
        if path.exists():
            annotations[key] = read_annotation(path)
    print(f"{len(annotations)} annotated files\n")

    header = (
        f"{'plugin':<12} {'stage':<6} {'n':>5} "
        f"{'F-meas':>7} {'CMLt':>6} {'AMLt':>6}"
    )
    print(header)
    print("-" * len(header))
    octave = {}
    for plugin in plugins:
        raw = read_beats_tsv(results / f"{plugin}.beats.tsv")
        scores, n = evaluate(raw, annotations)
        print(
            f"{plugin:<12} {'raw':<6} {n:>5} {scores['F-measure']:>7.3f} "
            f"{scores['CMLt']:>6.3f} {scores['AMLt']:>6.3f}"
        )
        grid_path = results / f"{plugin}.grid.tsv"
        if grid_path.exists():
            grids = read_grid_tsv(grid_path)
            scores, n = evaluate(
                {k: v[1] for k, v in grids.items()}, annotations
            )
            print(
                f"{plugin:<12} {'grid':<6} {n:>5} {scores['F-measure']:>7.3f} "
                f"{scores['CMLt']:>6.3f} {scores['AMLt']:>6.3f}"
            )
            counts = {}
            for key, (bpm, _) in grids.items():
                if key in annotations:
                    err = octave_error(bpm, annotations[key])
                    counts[err] = counts.get(err, 0) + 1
            octave[plugin] = counts

    if octave:
        print("\nGrid BPM vs. annotated tempo (4 % tolerance):")
        for plugin, counts in octave.items():
            total = sum(counts.values()) or 1
            summary = ", ".join(
                f"{k}: {v} ({100 * v / total:.1f} %)"
                for k, v in sorted(counts.items())
            )
            print(f"  {plugin:<12} {summary}")


if __name__ == "__main__":
    main()
