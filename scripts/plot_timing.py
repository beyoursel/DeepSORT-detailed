#!/usr/bin/env python3
"""Parse the per-frame timing CSV produced by YoloTracker (output.timing)
and render a boxplot of each pipeline stage's latency.

Dependencies: numpy + matplotlib only (no pandas/seaborn required).

Example:
    python3 scripts/plot_timing.py --input timing.csv --skip 3 \
        --output timing_boxplot.png
"""

import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

# Stages whose timing is nested inside another stage (e.g. DeepSORT's "reid"
# is measured inside "track"), so they must not be double-counted in the
# per-frame total.
NESTED_STAGES = {"reid": "track"}


def load_timing_csv(path):
    """Return (stage_names, {stage: np.ndarray of ms values})."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        stages = [c for c in (reader.fieldnames or []) if c != "frame"]
        if not stages:
            raise ValueError(f"no stage columns found in {path}")
        data = {s: [] for s in stages}
        n_rows = 0
        for row in reader:
            n_rows += 1
            for s in stages:
                v = row.get(s, "")
                data[s].append(float(v) if v else np.nan)
    if n_rows == 0:
        raise ValueError(f"no data rows in {path}")
    return stages, {s: np.array(v, dtype=float) for s, v in data.items()}


def print_summary(stages, data):
    header = f"{'stage':<12}{'mean':>10}{'std':>10}{'median':>10}" \
             f"{'p95':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for s in stages:
        v = data[s][~np.isnan(data[s])]
        if v.size == 0:
            continue
        print(f"{s:<12}{v.mean():>10.2f}{v.std():>10.2f}"
              f"{np.median(v):>10.2f}{np.percentile(v, 95):>10.2f}"
              f"{v.max():>10.2f}")


def plot_boxplot(stages, data, total_ms, output, title):
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * (len(stages) + 1)), 5))
    series = [data[s][~np.isnan(data[s])] for s in stages] + [total_ms]
    labels = stages + ["total"]
    ax.boxplot(series, tick_labels=labels, showmeans=True, whis=(5, 95))
    ax.set_ylabel("latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"boxplot saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="timing.csv",
                        help="timing CSV produced by output.timing")
    parser.add_argument("--skip", type=int, default=2,
                        help="drop the first N warmup frames (default: 2)")
    parser.add_argument("--output", default="timing_boxplot.png",
                        help="output image path")
    args = parser.parse_args()

    try:
        stages, data = load_timing_csv(args.input)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.skip > 0:
        for s in stages:
            data[s] = data[s][args.skip:]

    # Per-frame total = sum of top-level stages in that frame (NaN-safe);
    # nested stages (reid inside track) are excluded to avoid double counting.
    top_level = [s for s in stages if s not in NESTED_STAGES]
    total_ms = np.nansum(np.stack([data[s] for s in top_level]), axis=0)

    print(f"frames analyzed: {len(total_ms)} (skipped first {args.skip})\n")
    print_summary(stages, data)
    for child, parent in NESTED_STAGES.items():
        if child in stages and parent in stages:
            print(f"\nnote: \"{child}\" is nested inside \"{parent}\" and "
                  f"excluded from the frame total")
    print(f"\nframe total: mean={total_ms.mean():.2f} ms, "
          f"median={np.median(total_ms):.2f} ms, "
          f"p95={np.percentile(total_ms, 95):.2f} ms, "
          f"fps(mean)={1000.0 / total_ms.mean():.1f}")

    plot_boxplot(stages, data, total_ms, args.output,
                 title=f"Per-stage latency ({args.input})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
