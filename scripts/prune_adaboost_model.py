#!/usr/bin/env python3
"""Magnitude-prune an AdaBoost-format model file to its top-N |weight| features.

Keeps the N feature lines with the largest absolute weight (ties broken by
feature name for determinism) and recomputes nothing else -- the bias line
is copied through unchanged. Useful after upgrading a model to more features
than the original (e.g. the binary-perceptron-collapsed models from
scripts/collapse_binary_perceptron.py, issue #165) if the larger feature
count regresses inference throughput more than desired: quality typically
degrades gracefully down to a language-specific cliff, so sweep a few values
of N and check both held-out quality (`litsea evaluate`) and throughput
(`cargo bench -- external_corpus`) before picking one.

Usage: prune_adaboost_model.py <in.model> <out.model> <n>
"""
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <in.model> <out.model> <n>")
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                feat, w = line.split("\t")
                rows.append((feat, float(w)))
    rows.sort(key=lambda r: -abs(r[1]))

    with open(dst, "w", encoding="utf-8") as out:
        for feat, w in sorted(rows[:n]):
            out.write(f"{feat}\t{w}\n")
        out.write("0\n")

    print(f"{src}: {len(rows)} features -> {dst}: {min(n, len(rows))} features")


if __name__ == "__main__":
    main()
