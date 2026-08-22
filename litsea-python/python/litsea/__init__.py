"""Python binding for Litsea, a compact word segmentation and POS tagging library.

Models are not bundled with this package; download one from the Litsea
repository and load it by path, bytes, or URL::

    from litsea import Segmenter, Language

    seg = Segmenter.open(Language.JAPANESE, "models/japanese.model")
    seg.segment("すもももももももものうち")

The kind of model is detected from the file, so loading a ``*_pos.model``
gives a segmenter whose ``has_pos`` is ``True`` and whose
``segment_with_pos`` works.
"""

from ._litsea import (
    BinaryMetrics,
    CancelToken,
    Extractor,
    InvalidArgumentError,
    IoError,
    Language,
    LitseaError,
    ModelError,
    MulticlassMetrics,
    ParseError,
    PerceptronTrainer,
    PosUnavailableError,
    Segmenter,
    Token,
    Trainer,
    TwoStageMetrics,
    TwoStageTrainer,
    UnsupportedError,
    Upos,
    __version__,
    version,
)

__all__ = [
    "BinaryMetrics",
    "CancelToken",
    "Extractor",
    "InvalidArgumentError",
    "IoError",
    "Language",
    "LitseaError",
    "ModelError",
    "MulticlassMetrics",
    "ParseError",
    "PerceptronTrainer",
    "PosUnavailableError",
    "Segmenter",
    "Token",
    "Trainer",
    "TwoStageMetrics",
    "TwoStageTrainer",
    "UnsupportedError",
    "Upos",
    "__version__",
    "version",
]
