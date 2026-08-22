"""Error mapping: every failure raises a typed exception, never a panic."""

from __future__ import annotations

from pathlib import Path

import pytest
from litsea import (
    Extractor,
    InvalidArgumentError,
    IoError,
    Language,
    LitseaError,
    ModelError,
    ParseError,
    Segmenter,
    Trainer,
    TwoStageTrainer,
)


def test_unknown_language_name(models_dir: Path) -> None:
    """An unknown language names the supported ones in the message."""
    with pytest.raises(InvalidArgumentError) as excinfo:
        Segmenter.open("klingon", models_dir / "japanese.model")
    assert "klingon" in str(excinfo.value)
    assert "japanese" in str(excinfo.value)


def test_missing_model_file(tmp_path: Path) -> None:
    """A missing model file is an I/O error, not a crash."""
    with pytest.raises(IoError):
        Segmenter.open(Language.JAPANESE, tmp_path / "does-not-exist.model")


def test_malformed_model(tmp_path: Path) -> None:
    """A model that is not parseable is a parse error."""
    path = tmp_path / "broken.model"
    path.write_text("this is not a model\n")
    with pytest.raises(ParseError):
        Segmenter.open(Language.JAPANESE, path)


def test_empty_model(tmp_path: Path) -> None:
    """An empty model file is rejected."""
    path = tmp_path / "empty.model"
    path.write_text("")
    with pytest.raises(ParseError):
        Segmenter.open(Language.JAPANESE, path)


def test_legacy_joint_pos_model(tmp_path: Path) -> None:
    """Legacy joint POS models are rejected with actionable guidance."""
    path = tmp_path / "joint.model"
    # A bare integer first line is the joint class-count header.
    path.write_text("17\nfoo\t1.0\n")
    with pytest.raises(ModelError) as excinfo:
        Segmenter.open(Language.JAPANESE, path)
    assert "no longer supported" in str(excinfo.value)


def test_unknown_feature_set(tmp_path: Path) -> None:
    """An unknown two-stage feature set names the valid values."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT\n")
    with pytest.raises(InvalidArgumentError) as excinfo:
        Extractor(Language.JAPANESE).extract_two_stage(corpus, tmp_path / "features", feature_set="turbo")
    assert "turbo" in str(excinfo.value)


def test_missing_features_file(tmp_path: Path) -> None:
    """Training without a features file is an I/O error."""
    with pytest.raises(IoError):
        Trainer(0.01, 10, tmp_path / "missing.txt")


def test_dominance_out_of_range(tmp_path: Path) -> None:
    """A dominance outside (0.5, 1.0] is rejected before any work."""
    with pytest.raises((InvalidArgumentError, IoError)):
        TwoStageTrainer(1, tmp_path / "features", dominance=1.5)


def test_every_exception_derives_from_litsea_error(models_dir: Path) -> None:
    """One `except LitseaError` is enough to catch everything."""
    for error_type in (InvalidArgumentError, IoError, ModelError, ParseError):
        assert issubclass(error_type, LitseaError)

    with pytest.raises(LitseaError):
        Segmenter.open("klingon", models_dir / "japanese.model")
