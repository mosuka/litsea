#!/usr/bin/env python3
"""Collapse a 2-class (B/O) Averaged Perceptron model into scalar per-feature
weights in the existing AdaBoost model file format.

Context (issue #165): training a binary boundary classifier as an Averaged
Perceptron instead of AdaBoost reaches substantially higher held-out
segmentation quality on the same feature templates (see the pre-trained
models docs), because AdaBoost's presence-stump weak learners cannot express
the same decision boundaries a linear model can. The perceptron scores a
position purely as sum(matched-feature weights) per class (there is no
perceptron-level bias term), so score_B - score_O = sum(matched
(w_B[f] - w_O[f])) exactly. Writing "feat\\tweight" lines (weight = w_B - w_O,
skipping zero) plus a literal "0" bias line, then loading the result with
litsea's existing AdaBoost loader, reproduces this: the AdaBoost format
defines bias() to equal the written bias line verbatim regardless of the
feature weights (the algebraic inverse of AdaBoost's own bias computation),
so the collapsed model's `score >= 0.0` decision becomes exactly
`score_B >= score_O` -- including the tie case, which both the perceptron's
first-wins rule ("B" sorts before "O", so B is class index 0) and this
comparison resolve to B (boundary). This is the same derivation used for
two-stage stage-1 training (see litsea/src/trainer.rs's
collapse_boundary_perceptron, issue #168); this script performs the
identical transform outside the crate, so bundled segmentation models can be
upgraded with zero engine/code changes.

Usage:
    litsea extract -l <language> [--format tsv] <corpus> <features.txt>
    # remap boundary labels 1/-1 -> B/O (tie-break correctness -- see above)
    sed -i 's/^1\\t/B\\t/; s/^-1\\t/O\\t/' <features.txt>
    litsea train --perceptron --num-epochs <N> <features.txt> <perceptron.model>
    scripts/collapse_binary_perceptron.py <perceptron.model> <out.model>
"""
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <perceptron.model> <out.model>")
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, encoding="utf-8") as f:
        n = int(f.readline())
        classes = [f.readline().rstrip("\n") for _ in range(n)]
        if classes != ["B", "O"]:
            sys.exit(
                f"error: expected a 2-class model with classes ['B', 'O'], got {classes}\n"
                "(remap boundary labels 1/-1 -> B/O before training -- see this script's docstring)"
            )
        weights = {}
        for line in f:
            feat, cls, w = line.rstrip("\n").split("\t")
            weights.setdefault(feat, {})[cls] = float(w)

    collapsed = {}
    for feat, cw in weights.items():
        w = cw.get("B", 0.0) - cw.get("O", 0.0)
        if w != 0.0:
            collapsed[feat] = w

    with open(dst, "w", encoding="utf-8") as out:
        for feat in sorted(collapsed):
            out.write(f"{feat}\t{collapsed[feat]}\n")
        out.write("0\n")

    print(f"{src}: {len(weights)} features -> {dst}: {len(collapsed)} non-zero features")


if __name__ == "__main__":
    main()
