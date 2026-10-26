"""Segmentation, POS tagging, and enum behaviour."""

from __future__ import annotations

import enum
import threading
from pathlib import Path

import pytest
from conftest import run_cli
from litsea import InvalidArgumentError, Language, PosUnavailableError, Segmenter, Token, Upos

# One sentence per language, with the segmentation model that handles it.
SEGMENTATION_CASES = [
    ("japanese", "japanese.model", "これはテストです。"),
    ("chinese", "chinese.model", "我喜欢吃中国菜。"),
    ("korean", "korean.model", "안녕하세요 반갑습니다"),
    ("english", "english.model", "The quick brown fox jumps over the lazy dog."),
]

POS_CASES = [
    ("japanese", "japanese_pos.model", "これはテストです。"),
    ("korean", "korean_pos.model", "안녕하세요 반갑습니다"),
]

# User dictionary cases (#263): the model splits each registered word
# without the dictionary, and the English word contains a space.
USER_WORD_CASES = [
    ("japanese", "japanese.model", ["雪だるま", "新幹線"], "雪だるまを作る"),
    ("japanese", "japanese.model", ["雪だるま", "新幹線"], "新幹線で帰る"),
    ("english", "english.model", ["New York"], "I live in New York."),
]


def write_words(directory: Path, words: list[str]) -> Path:
    """Write a `--user-words` file for the CLI and return its path."""
    path = directory / "words.txt"
    path.write_text("".join(word + "\n" for word in words), encoding="utf-8")
    return path


@pytest.mark.parametrize(("language", "model", "sentence"), SEGMENTATION_CASES)
def test_segment_matches_the_cli(models_dir: Path, litsea_cli: Path, language: str, model: str, sentence: str) -> None:
    """The binding must produce exactly what `litsea segment` produces.

    The comparison is against the CLI's rendered line rather than a re-split
    of it: the CLI joins tokens with a space, so for space-preserving
    languages (Korean, English) a whitespace token cannot be recovered by
    splitting the output again.
    """
    expected = run_cli(
        litsea_cli,
        ["segment", "-l", language, str(models_dir / model)],
        sentence + "\n",
    )[0]

    seg = Segmenter.open(language, models_dir / model)
    assert " ".join(seg.segment(sentence)) == expected


@pytest.mark.parametrize(("language", "model", "sentence"), POS_CASES)
def test_segment_with_pos_matches_the_cli(
    models_dir: Path, litsea_cli: Path, language: str, model: str, sentence: str
) -> None:
    """POS output must match `litsea segment --pos`, rendered the same way."""
    expected = run_cli(
        litsea_cli,
        ["segment", "-l", language, "--pos", str(models_dir / model)],
        sentence + "\n",
    )[0]

    seg = Segmenter.open(language, models_dir / model)
    actual = " ".join(f"{token.surface}/{token.pos.name}" for token in seg.segment_with_pos(sentence))
    assert actual == expected


@pytest.mark.parametrize(("language", "model", "words", "sentence"), USER_WORD_CASES)
def test_user_words_match_the_cli(
    models_dir: Path,
    litsea_cli: Path,
    tmp_path: Path,
    language: str,
    model: str,
    words: list[str],
    sentence: str,
) -> None:
    """`user_words` must produce exactly what `litsea segment --user-words` does."""
    expected = run_cli(
        litsea_cli,
        ["segment", "-l", language, "--user-words", str(write_words(tmp_path, words)), str(models_dir / model)],
        sentence + "\n",
    )[0]

    seg = Segmenter.open(language, models_dir / model, user_words=words)
    assert " ".join(seg.segment(sentence)) == expected
    # The dictionary must actually matter for these cases.
    assert seg.segment(sentence) != Segmenter.open(language, models_dir / model).segment(sentence)


def test_user_words_with_pos_match_the_cli(models_dir: Path, litsea_cli: Path, tmp_path: Path) -> None:
    """A dictionary word is one tagged token, as with `litsea segment --pos --user-words`."""
    sentence = "雪だるまを作る"
    words = ["雪だるま"]
    expected = run_cli(
        litsea_cli,
        [
            "segment",
            "-l",
            "japanese",
            "--pos",
            "--user-words",
            str(write_words(tmp_path, words)),
            str(models_dir / "japanese_pos.model"),
        ],
        sentence + "\n",
    )[0]

    seg = Segmenter.open("japanese", models_dir / "japanese_pos.model", user_words=words)
    actual = " ".join(f"{token.surface}/{token.pos.name}" for token in seg.segment_with_pos(sentence))
    assert actual == expected
    assert seg.segment_with_pos(sentence)[0].surface == "雪だるま"


def test_user_words_on_every_constructor(models_dir: Path) -> None:
    """`open`, `from_bytes`, and `from_uri` all accept the dictionary."""
    path = models_dir / "japanese.model"
    sentence = "雪だるまを作る"
    words = ("雪だるま",)  # any sequence of strings works

    from_path = Segmenter.open(Language.JAPANESE, path, user_words=words)
    from_bytes = Segmenter.from_bytes(Language.JAPANESE, path.read_bytes(), user_words=words)
    from_uri = Segmenter.from_uri(Language.JAPANESE, str(path), user_words=words)

    assert from_path.segment(sentence) == ["雪だるま", "を", "作る"]
    assert from_bytes.segment(sentence) == from_path.segment(sentence)
    assert from_uri.segment(sentence) == from_path.segment(sentence)


def test_user_words_gate(models_dir: Path) -> None:
    """Gate 0 applies a word only where the model already agrees with it."""
    path = models_dir / "japanese.model"
    sentence = "雪だるまを作る"
    plain = Segmenter.open("japanese", path).segment(sentence)

    gated = Segmenter.open("japanese", path, user_words=["雪だるま"], user_words_gate=0.0)
    assert gated.segment(sentence) == plain
    open_gate = Segmenter.open("japanese", path, user_words=["雪だるま"], user_words_gate=1.0)
    assert open_gate.segment(sentence) == ["雪だるま", "を", "作る"]
    # An empty list is no dictionary at all.
    assert Segmenter.open("japanese", path, user_words=[]).segment(sentence) == plain


def test_user_words_invalid_arguments(models_dir: Path) -> None:
    """A gate outside 0..1 is an `InvalidArgumentError`, words must be strings."""
    path = models_dir / "japanese.model"
    for words in (["雪だるま"], None):
        with pytest.raises(InvalidArgumentError, match=r"0\.0\.\.=1\.0"):
            Segmenter.open("japanese", path, user_words=words, user_words_gate=1.5)
    with pytest.raises(TypeError):
        Segmenter.open("japanese", path, user_words="雪だるま")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Segmenter.open("japanese", path, ["雪だるま"])  # type: ignore[misc]


# Tagged user dictionary entries (#264): the tagger gives both words NOUN
# without a tag (`ポケモン` through the lexicon, `チャットGPT` through the full
# argmax).
USER_WORD_TAGS = {"チャットGPT": "PROPN", "ポケモン": "PROPN"}
TAGGED_SENTENCES = ["チャットGPTに聞く", "ポケモンを集める", "雪だるまを作る"]


def tagged(segmenter: Segmenter, sentence: str) -> str:
    """Render `segment_with_pos` the way `litsea segment --pos` prints it."""
    return " ".join(f"{token.surface}/{token.pos.name}" for token in segmenter.segment_with_pos(sentence))


def test_user_word_tags_match_the_cli(models_dir: Path, litsea_cli: Path, tmp_path: Path) -> None:
    """`user_word_tags` must produce what a tagged `--user-words` file does."""
    model = models_dir / "japanese_pos.model"
    words_file = tmp_path / "words.txt"
    words_file.write_text(
        "".join(f"{word}\t{tag}\n" for word, tag in USER_WORD_TAGS.items()) + "雪だるま\n",
        encoding="utf-8",
    )
    expected = run_cli(
        litsea_cli,
        ["segment", "-l", "japanese", "--pos", "--user-words", str(words_file), str(model)],
        "".join(sentence + "\n" for sentence in TAGGED_SENTENCES),
    )

    seg = Segmenter.open("japanese", model, user_words=["雪だるま"], user_word_tags=USER_WORD_TAGS)
    assert [tagged(seg, sentence) for sentence in TAGGED_SENTENCES] == expected
    assert tagged(seg, "チャットGPTに聞く").startswith("チャットGPT/PROPN ")
    # The tags must actually matter: untagged, the same words come out as NOUN.
    untagged = Segmenter.open("japanese", model, user_words=["雪だるま", *USER_WORD_TAGS])
    assert tagged(untagged, "チャットGPTに聞く").startswith("チャットGPT/NOUN ")
    assert tagged(seg, "ポケモンを集める") != tagged(untagged, "ポケモンを集める")


def test_user_word_tags_on_every_constructor(models_dir: Path) -> None:
    """`open`, `from_bytes`, and `from_uri` all accept tags, also without `user_words`."""
    path = models_dir / "japanese_pos.model"
    sentence = "チャットGPTに聞く"

    from_path = Segmenter.open(Language.JAPANESE, path, user_word_tags=USER_WORD_TAGS)
    from_bytes = Segmenter.from_bytes(Language.JAPANESE, path.read_bytes(), user_word_tags=USER_WORD_TAGS)
    from_uri = Segmenter.from_uri(Language.JAPANESE, str(path), user_word_tags=USER_WORD_TAGS)

    tokens = from_path.segment_with_pos(sentence)
    assert (tokens[0].surface, tokens[0].pos) == ("チャットGPT", Upos.PROPN)
    assert tagged(from_bytes, sentence) == tagged(from_path, sentence)
    assert tagged(from_uri, sentence) == tagged(from_path, sentence)
    # Segmentation alone ignores the tags.
    assert from_path.segment(sentence) == ["チャットGPT", "に", "聞く"]


def test_user_word_tags_invalid_arguments(models_dir: Path) -> None:
    """A tag that is not UPOS is an `InvalidArgumentError`; the tags must be a dict of strings."""
    path = models_dir / "japanese_pos.model"
    with pytest.raises(InvalidArgumentError, match=r'unknown UPOS tag "Noun"'):
        Segmenter.open("japanese", path, user_word_tags={"ポケモン": "Noun"})
    with pytest.raises(InvalidArgumentError, match=r"0\.0\.\.=1\.0"):
        Segmenter.open("japanese", path, user_word_tags=USER_WORD_TAGS, user_words_gate=1.5)
    with pytest.raises(TypeError):
        Segmenter.open("japanese", path, user_word_tags=["ポケモン"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Segmenter.open("japanese", path, user_word_tags={"ポケモン": 1})  # type: ignore[dict-item]


@pytest.mark.parametrize(("language", "model", "sentence"), SEGMENTATION_CASES)
def test_offsets_reconstruct_the_input(models_dir: Path, language: str, model: str, sentence: str) -> None:
    """Byte offsets must tile the input exactly, with no gaps."""
    seg = Segmenter.open(language, models_dir / model)
    tokens = seg.segment_tokens(sentence)
    raw = sentence.encode()

    assert tokens
    expected_start = 0
    for token in tokens:
        assert token.start == expected_start
        assert raw[token.start : token.end].decode() == token.surface
        assert token.pos is None
        expected_start = token.end
    assert expected_start == len(raw)
    assert "".join(token.surface for token in tokens) == sentence


def test_whitespace_is_its_own_token(models_dir: Path) -> None:
    """Space-delimited languages keep the space as a token."""
    seg = Segmenter.open(Language.KOREAN, models_dir / "korean.model")
    assert seg.segment("안녕하세요 반갑습니다") == ["안녕하세요", " ", "반갑습니다"]


def test_batch_matches_single_calls(models_dir: Path) -> None:
    """`segment_batch` must agree with repeated `segment` calls."""
    seg = Segmenter.open(Language.JAPANESE, models_dir / "japanese.model")
    sentences = ["これはテストです。", "", "東京都から神奈川県へ引っ越した"]

    assert seg.segment_batch(sentences) == [seg.segment(s) for s in sentences]
    assert seg.segment_batch(sentences)[1] == []


def test_pos_batch_matches_single_calls(models_dir: Path) -> None:
    """`segment_with_pos_batch` must agree with repeated calls."""
    seg = Segmenter.open(Language.JAPANESE, models_dir / "japanese_pos.model")
    sentences = ["これはテストです。", "東京都から神奈川県へ引っ越した"]

    batched = seg.segment_with_pos_batch(sentences)
    assert batched == [seg.segment_with_pos(s) for s in sentences]


def test_model_kind_is_detected(models_dir: Path) -> None:
    """`has_pos` reflects the model that was loaded, with no flag passed."""
    assert not Segmenter.open("ja", models_dir / "japanese.model").has_pos
    assert Segmenter.open("ja", models_dir / "japanese_pos.model").has_pos


def test_pos_on_segmentation_model_raises(models_dir: Path) -> None:
    """POS tagging a segmentation-only model is a typed error."""
    seg = Segmenter.open(Language.JAPANESE, models_dir / "japanese.model")
    with pytest.raises(PosUnavailableError, match="two-stage POS model"):
        seg.segment_with_pos("これはテストです。")


def test_loading_sources_agree(models_dir: Path) -> None:
    """`open`, `from_bytes`, and `from_uri` produce the same segmenter."""
    path = models_dir / "japanese.model"
    sentence = "これはテストです。"

    from_path = Segmenter.open(Language.JAPANESE, path)
    from_bytes = Segmenter.from_bytes(Language.JAPANESE, path.read_bytes())
    from_uri = Segmenter.from_uri(Language.JAPANESE, str(path))

    assert from_path.segment(sentence) == from_bytes.segment(sentence)
    assert from_path.segment(sentence) == from_uri.segment(sentence)


def test_language_argument_accepts_strings(models_dir: Path) -> None:
    """A `Language` member and its names are interchangeable."""
    path = models_dir / "japanese.model"
    sentence = "これはテストです。"

    expected = Segmenter.open(Language.JAPANESE, path).segment(sentence)
    for name in ("ja", "JA", "japanese", "Japanese"):
        assert Segmenter.open(name, path).segment(sentence) == expected


def test_segmenter_properties(models_dir: Path) -> None:
    """`language` and `repr` report the loaded configuration."""
    seg = Segmenter.open(Language.KOREAN, models_dir / "korean.model")
    assert seg.language == Language.KOREAN
    assert seg.language.name == "korean"
    assert "korean" in repr(seg)


def test_shared_between_threads(models_dir: Path) -> None:
    """One segmenter can serve several threads."""
    seg = Segmenter.open(Language.JAPANESE, models_dir / "japanese.model")
    sentence = "これはテストです。"
    expected = seg.segment(sentence)
    results: list[list[str]] = []
    lock = threading.Lock()

    def work() -> None:
        result = seg.segment_batch([sentence] * 50)
        with lock:
            results.extend(result)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 200
    assert all(result == expected for result in results)


def test_language_enum() -> None:
    """`Language` exposes its members, names, and parsing."""
    assert Language.parse("ko") == Language.KOREAN
    assert Language.KOREAN.name == "korean"
    assert str(Language.ENGLISH) == "english"
    assert [language.name for language in Language.all()] == [
        "japanese",
        "chinese",
        "korean",
        "english",
    ]


def test_enums_are_not_python_enums() -> None:
    """The members are class attributes, as the type stubs describe.

    PyO3 classes are not `enum.Enum` subclasses, so iterating the class
    raises; `all()` is the supported way to enumerate. Pinning this keeps
    the stubs honest.
    """
    assert not isinstance(Language.JAPANESE, enum.Enum)
    with pytest.raises(TypeError):
        list(Language)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        Language["JAPANESE"]  # type: ignore[index]


def test_enum_members_do_not_compare_equal_to_ints() -> None:
    """Without `eq_int`, discriminants never leak into comparisons."""
    assert Language.JAPANESE != 0
    assert Upos.ADJ != 0
    assert Language.JAPANESE == Language.JAPANESE
    assert Language.JAPANESE != Language.CHINESE


def test_upos_enum() -> None:
    """`Upos` exposes all 17 UD tags."""
    assert len(Upos.all()) == 17
    assert Upos.NOUN.name == "NOUN"
    assert str(Upos.VERB) == "VERB"
    assert Upos.NOUN != Upos.VERB


def test_token_repr_and_equality(models_dir: Path) -> None:
    """Tokens compare by value and print readably."""
    seg = Segmenter.open(Language.JAPANESE, models_dir / "japanese_pos.model")
    first, second = (seg.segment_with_pos("これはテストです。") for _ in range(2))

    assert first == second
    assert isinstance(first[0], Token)
    assert "pos=" in repr(first[0])
