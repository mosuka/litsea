"""Train a segmentation model, cancelling it from another thread.

Usage:
    python examples/train.py corpus.txt out.model

The corpus is one sentence per line, with words separated by spaces:

    これ は テスト です 。
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

from litsea import CancelToken, Extractor, Language, Trainer


def main() -> int:
    """Run the example.

    Returns:
        0 on success, 2 when the arguments are wrong.
    """
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    corpus, model = Path(sys.argv[1]), Path(sys.argv[2])

    with tempfile.TemporaryDirectory() as tmp:
        features = Path(tmp) / "features.txt"

        print(f"extracting features from {corpus} ...")
        Extractor(Language.JAPANESE).extract(corpus, features)

        # Training releases the GIL, so a timer thread can stop it. Cancelling
        # is not an error: the partially trained model is still written.
        cancel = CancelToken()
        timer = threading.Timer(60.0, cancel.cancel)
        timer.start()

        print("training (will stop after 60s if it has not converged) ...")
        try:
            metrics = Trainer(0.01, 10_000, features).train(model, cancel=cancel)
        finally:
            timer.cancel()

    print(f"wrote {model}")
    print(f"  accuracy:  {metrics.accuracy:.2f}%")
    print(f"  precision: {metrics.precision:.2f}%")
    print(f"  recall:    {metrics.recall:.2f}%")
    print(f"  instances: {metrics.num_instances}")
    if cancel.cancelled:
        print("  (training was cancelled; the model is partially trained)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
