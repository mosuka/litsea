"""Segment a sentence, with and without POS tags.

Usage:
    python examples/segment.py ../models/japanese.model "これはテストです。"
    python examples/segment.py ../models/japanese_pos.model "これはテストです。"
"""

from __future__ import annotations

import sys

from litsea import Segmenter


def main() -> int:
    """Run the example.

    Returns:
        0 on success, 2 when the arguments are wrong.
    """
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    model_path, text = sys.argv[1], sys.argv[2]

    # The model file identifies its own kind, so nothing here says "this is
    # a POS model" - `has_pos` reports what was loaded.
    segmenter = Segmenter.open("japanese", model_path)
    print(f"model: {model_path} (has_pos={segmenter.has_pos})")

    print("tokens:", " ".join(segmenter.segment(text)))

    if segmenter.has_pos:
        print("tagged:")
        for token in segmenter.segment_with_pos(text):
            print(f"  {token.surface}\t{token.pos.name}\t[{token.start}:{token.end}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
