"""Feature extraction, training, and cancellation."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from conftest import run_cli
from litsea import (
    CancelToken,
    Extractor,
    InvalidArgumentError,
    Language,
    PerceptronTrainer,
    Segmenter,
    Trainer,
    TwoStageTrainer,
)

SENTENCES = [
    "これ は テスト です 。",
    "隣 の 客 は よく 柿 食う 客 だ",
    "東京 都 から 神奈川 県 へ 引っ越し た",
]

POS_SENTENCES = [
    "これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT",
    "隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX",
    "東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX",
]


def write_corpus(path: Path, sentences: list[str], repeats: int = 20) -> Path:
    """Write a small corpus, repeated so training has something to learn."""
    path.write_text("\n".join(sentences * repeats) + "\n")
    return path


def test_extract_then_train_round_trip(tmp_path: Path, litsea_cli: Path) -> None:
    """A model trained from Python must load in the Rust CLI."""
    corpus = write_corpus(tmp_path / "corpus.txt", SENTENCES)
    features = tmp_path / "features.txt"
    model = tmp_path / "trained.model"

    Extractor(Language.JAPANESE).extract(corpus, features)
    assert features.exists()

    metrics = Trainer(0.01, 20, features).train(model)
    assert metrics.num_instances > 0
    assert 0.0 <= metrics.accuracy <= 100.0
    assert "accuracy" in repr(metrics)

    # The CLI is the independent check that the file is a valid model.
    output = run_cli(litsea_cli, ["segment", "-l", "japanese", str(model)], "これはテストです。\n")
    assert output and output[0]

    # And the binding agrees with the CLI on the freshly trained model.
    assert Segmenter.open(Language.JAPANESE, model).segment("これはテストです。") == output[0].split(" ")


def test_tag_free_extraction_is_smaller(tmp_path: Path) -> None:
    """`tag_free=True` drops the tag-dependent templates."""
    corpus = write_corpus(tmp_path / "corpus.txt", SENTENCES)
    extractor = Extractor(Language.JAPANESE)

    full = tmp_path / "full.txt"
    lean = tmp_path / "lean.txt"
    extractor.extract(corpus, full)
    extractor.extract(corpus, lean, tag_free=True)

    assert lean.stat().st_size < full.stat().st_size


def test_two_stage_training_round_trip(tmp_path: Path) -> None:
    """Two-stage training produces a POS-capable model."""
    corpus = write_corpus(tmp_path / "corpus_pos.txt", POS_SENTENCES)
    prefix = tmp_path / "features"
    model = tmp_path / "two_stage.model"

    Extractor(Language.JAPANESE).extract_two_stage(corpus, prefix, feature_set="fast")
    for suffix in ("stage1", "stage2", "lexicon"):
        assert (tmp_path / f"features.{suffix}").exists()

    trainer = TwoStageTrainer(3, prefix)
    assert trainer.available
    metrics = trainer.train(model)
    assert metrics.stage1.num_instances > 0
    assert metrics.stage2.num_instances > 0

    seg = Segmenter.open(Language.JAPANESE, model)
    assert seg.has_pos
    tokens = seg.segment_with_pos("これはテストです。")
    assert tokens
    assert all(token.pos is not None for token in tokens)


def test_two_stage_trainer_cannot_be_reused(tmp_path: Path) -> None:
    """The trainer is consumed by training, and says so."""
    corpus = write_corpus(tmp_path / "corpus_pos.txt", POS_SENTENCES)
    prefix = tmp_path / "features"
    model = tmp_path / "two_stage.model"

    Extractor(Language.JAPANESE).extract_two_stage(corpus, prefix)
    trainer = TwoStageTrainer(1, prefix)
    trainer.train(model)

    assert not trainer.available
    with pytest.raises(InvalidArgumentError, match="already been used"):
        trainer.train(model)


def test_perceptron_trainer(tmp_path: Path) -> None:
    """The perceptron trainer trains from stage-2 features."""
    corpus = write_corpus(tmp_path / "corpus_pos.txt", POS_SENTENCES)
    prefix = tmp_path / "features"
    model = tmp_path / "perceptron.model"

    Extractor(Language.JAPANESE).extract_two_stage(corpus, prefix)
    metrics = PerceptronTrainer(2, tmp_path / "features.stage2").train(model)

    assert metrics.num_instances > 0
    assert metrics.gold_per_class
    assert model.exists()


def test_cancel_before_training_still_writes_a_model(tmp_path: Path) -> None:
    """Cancelling is cooperative: it is not an error."""
    corpus = write_corpus(tmp_path / "corpus.txt", SENTENCES)
    features = tmp_path / "features.txt"
    model = tmp_path / "cancelled.model"

    Extractor(Language.JAPANESE).extract(corpus, features)

    cancel = CancelToken()
    cancel.cancel()
    assert cancel.cancelled

    metrics = Trainer(0.01, 100_000, features).train(model, cancel=cancel)
    assert metrics.num_instances > 0
    assert model.exists()


def test_training_releases_the_gil(tmp_path: Path) -> None:
    """Another Python thread must be able to run while training does.

    If `train` held the GIL, the canceller thread could not execute until
    training returned, so its timestamp would land after training finished.

    The iteration count is bounded (~35 ms per iteration on this corpus, so
    ~7 s uncancelled) so that a regression fails the assertion instead of
    hanging the suite.
    """
    corpus = write_corpus(tmp_path / "corpus.txt", SENTENCES, repeats=400)
    features = tmp_path / "features.txt"
    model = tmp_path / "cancelled.model"

    Extractor(Language.JAPANESE).extract(corpus, features)

    cancel = CancelToken()
    started = threading.Event()
    cancelled_at: list[float] = []

    def canceller() -> None:
        started.wait(timeout=30.0)
        time.sleep(0.2)
        cancelled_at.append(time.monotonic())
        cancel.cancel()

    thread = threading.Thread(target=canceller)
    thread.start()

    trainer = Trainer(0.0, 200, features)
    started.set()
    train_started = time.monotonic()
    metrics = trainer.train(model, cancel=cancel)
    train_finished = time.monotonic()
    thread.join(timeout=30.0)

    assert cancelled_at, "the canceller thread never ran"
    assert train_started < cancelled_at[0] < train_finished, (
        "the canceller ran outside the training window, so the GIL was not released"
    )
    assert metrics.num_instances > 0
    assert model.exists()
