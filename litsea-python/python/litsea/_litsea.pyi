"""Type stubs for the compiled `litsea._litsea` extension module."""

import os
from collections.abc import Mapping, Sequence
from typing import ClassVar, TypeAlias

__version__: str

def version() -> str:
    """Return the version of the underlying ``litsea`` crate."""

class Language:
    """A language supported by Litsea's models.

    Anywhere a ``Language`` is accepted, its name works too (``"japanese"``
    or ``"ja"``, case-insensitive).

    This is a PyO3 class rather than a :class:`enum.Enum`: the members are
    class attributes, so ``for language in Language`` does not work. Use
    :meth:`Language.all` to enumerate them. ``str(language)`` returns the
    canonical name.
    """

    JAPANESE: ClassVar[Language]
    CHINESE: ClassVar[Language]
    KOREAN: ClassVar[Language]
    ENGLISH: ClassVar[Language]

    @property
    def name(self) -> str:
        """The canonical lowercase name, for example ``"japanese"``."""

    @staticmethod
    def parse(name: str) -> Language:
        """Parse a language name or ISO 639-1 code."""

    @staticmethod
    def all() -> list[Language]:
        """Return every supported language."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Upos:
    """A Universal POS tag.

    Like :class:`Language`, this is a PyO3 class rather than a
    :class:`enum.Enum`; use :meth:`Upos.all` to enumerate the tags.
    ``str(tag)`` returns the tag name.
    """

    ADJ: ClassVar[Upos]
    ADP: ClassVar[Upos]
    ADV: ClassVar[Upos]
    AUX: ClassVar[Upos]
    CCONJ: ClassVar[Upos]
    DET: ClassVar[Upos]
    INTJ: ClassVar[Upos]
    NOUN: ClassVar[Upos]
    NUM: ClassVar[Upos]
    PART: ClassVar[Upos]
    PRON: ClassVar[Upos]
    PROPN: ClassVar[Upos]
    PUNCT: ClassVar[Upos]
    SCONJ: ClassVar[Upos]
    SYM: ClassVar[Upos]
    VERB: ClassVar[Upos]
    X: ClassVar[Upos]

    @property
    def name(self) -> str:
        """The tag name, for example ``"NOUN"``."""

    @staticmethod
    def all() -> list[Upos]:
        """Return all 17 tags, in UD order."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Token:
    """A segmented token.

    ``start`` and ``end`` are byte offsets into the input string.
    """

    @property
    def surface(self) -> str: ...
    @property
    def pos(self) -> Upos | None: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...

LanguageArg: TypeAlias = Language | str

class Segmenter:
    """A word segmenter, optionally with POS tagging."""

    @staticmethod
    def open(language: LanguageArg, path: str | os.PathLike[str]) -> Segmenter:
        """Load a model from a filesystem path."""

    @staticmethod
    def from_bytes(language: LanguageArg, data: bytes) -> Segmenter:
        """Load a model from raw bytes."""

    @staticmethod
    def from_uri(language: LanguageArg, uri: str) -> Segmenter:
        """Load a model from a path, ``file://`` path, or ``http(s)://`` URL."""

    @property
    def language(self) -> Language: ...
    @property
    def has_pos(self) -> bool: ...
    def segment(self, text: str) -> list[str]:
        """Split a sentence into tokens."""

    def segment_batch(self, texts: Sequence[str]) -> list[list[str]]:
        """Split several sentences into tokens, releasing the GIL."""

    def segment_tokens(self, text: str) -> list[Token]:
        """Split a sentence into tokens carrying byte offsets."""

    def segment_with_pos(self, text: str) -> list[Token]:
        """Split a sentence into tokens and tag each with a UPOS tag."""

    def segment_with_pos_batch(self, texts: Sequence[str]) -> list[list[Token]]:
        """Split and tag several sentences, releasing the GIL."""

class CancelToken:
    """A flag that asks a running training job to stop."""

    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...

class BinaryMetrics:
    """Metrics from training a binary (segmentation) model."""

    @property
    def accuracy(self) -> float: ...
    @property
    def precision(self) -> float: ...
    @property
    def recall(self) -> float: ...
    @property
    def num_instances(self) -> int: ...
    @property
    def true_positives(self) -> int: ...
    @property
    def false_positives(self) -> int: ...
    @property
    def false_negatives(self) -> int: ...
    @property
    def true_negatives(self) -> int: ...

class MulticlassMetrics:
    """Metrics from training a multiclass model."""

    @property
    def accuracy(self) -> float: ...
    @property
    def macro_precision(self) -> float: ...
    @property
    def macro_recall(self) -> float: ...
    @property
    def num_instances(self) -> int: ...
    @property
    def correct_per_class(self) -> Mapping[str, int]: ...
    @property
    def predicted_per_class(self) -> Mapping[str, int]: ...
    @property
    def gold_per_class(self) -> Mapping[str, int]: ...

class TwoStageMetrics:
    """Metrics from training a two-stage model."""

    @property
    def stage1(self) -> MulticlassMetrics: ...
    @property
    def stage2(self) -> MulticlassMetrics: ...

class Extractor:
    """Extracts training features from a corpus."""

    def __init__(self, language: LanguageArg) -> None: ...
    def extract(
        self,
        corpus_path: str | os.PathLike[str],
        features_path: str | os.PathLike[str],
        *,
        tsv: bool = False,
        tag_free: bool = False,
    ) -> None: ...
    def extract_two_stage(
        self,
        corpus_path: str | os.PathLike[str],
        output_prefix: str | os.PathLike[str],
        *,
        feature_set: str = "fast",
        tsv: bool = False,
    ) -> None: ...

class Trainer:
    """Trains a segmentation model."""

    def __init__(
        self,
        threshold: float,
        num_iterations: int,
        features_path: str | os.PathLike[str],
    ) -> None: ...
    def load_model(self, model_uri: str) -> None: ...
    def train(
        self,
        model_path: str | os.PathLike[str],
        *,
        cancel: CancelToken | None = None,
    ) -> BinaryMetrics: ...

class PerceptronTrainer:
    """Trains a label-agnostic Averaged Perceptron model."""

    def __init__(self, num_epochs: int, features_path: str | os.PathLike[str]) -> None: ...
    def load_model(self, model_uri: str) -> None: ...
    def train(
        self,
        model_path: str | os.PathLike[str],
        *,
        cancel: CancelToken | None = None,
    ) -> MulticlassMetrics: ...

class TwoStageTrainer:
    """Trains a two-stage segmentation + POS model.

    A trainer can only be used once; check ``available`` before reusing one.
    """

    def __init__(
        self,
        num_epochs: int,
        features_prefix: str | os.PathLike[str],
        *,
        dominance: float = 0.99,
    ) -> None: ...
    @property
    def available(self) -> bool: ...
    def train(
        self,
        model_path: str | os.PathLike[str],
        *,
        cancel: CancelToken | None = None,
    ) -> TwoStageMetrics: ...

class LitseaError(Exception):
    """Base class for every error raised by litsea."""

class InvalidArgumentError(LitseaError):
    """An argument was invalid, such as an unknown language name."""

class ModelError(LitseaError):
    """A model could not be obtained, or is not the kind the call needs."""

class IoError(LitseaError):
    """A file could not be read or written."""

class ParseError(LitseaError):
    """A model or training data file is malformed."""

class UnsupportedError(LitseaError):
    """The operation is not supported in this build or environment."""

class PosUnavailableError(LitseaError):
    """POS tagging was requested from a segmentation-only model."""
