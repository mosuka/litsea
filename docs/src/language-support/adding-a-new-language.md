# Adding a New Language

Litsea's multilingual framework is designed to be easily extensible. This guide explains how to add support for a new language, using the addition of English (issue #194) as the worked example throughout.

## Steps Overview

1. Add a variant to the `Language` enum
2. Implement `Display` and `FromStr` match arms
3. Create a character classification function
4. Register the classification function
5. Decide on WC feature inclusion
6. Choose a corpus protocol (space-separated or space-preserving TSV)
7. Train the bundled segmentation model (binary-perceptron-collapse recipe)
8. Optionally train a two-stage POS model
9. Add held-out evaluation gold files
10. Add tests

## Step 1: Add a Variant to `Language`

In `litsea/src/language.rs`, add a new variant to the `Language` enum:

```rust
#[non_exhaustive]
pub enum Language {
    #[default]
    Japanese,
    Chinese,
    Korean,
    English,
    Thai,       // ← new language
}
```

The enum is marked `#[non_exhaustive]` precisely because new languages are expected to be added, so adding a variant is not a breaking change for downstream crates.

## Step 2: Implement Display and FromStr

Add match arms for the new language:

```rust
// In Display impl
Language::Thai => write!(f, "thai"),

// In FromStr impl
"thai" | "th" => Ok(Language::Thai),
```

Also update the `ParseLanguageError` message in `language.rs`: it enumerates the supported languages (`Supported: japanese (ja), chinese (zh), korean (ko), english (en)`) and is pinned by a unit test, so both the message and the test must include the new language.

## Step 3: Create a Character Classification Function

Define a function that classifies a `char` into a **type id** for the new language. Ids are indices into the language's ordered `type_codes()` table (Step 4): the shared classes occupy fixed indices ("O" = 0, "P" = 1, "A" = 2, "N" = 3) and language-specific classes follow from 4. Classification is a direct `match` on character ranges (no regex), so each class is an arm; the **first matching arm wins**:

```rust
fn thai_char_type_id(c: char) -> u8 {
    match c {
        // Thai consonants and sequential vowels (U+0E01-U+0E3A)
        '\u{0E01}'..='\u{0E3A}' => 4, // "T"
        // Thai vowels and tone marks (U+0E40-U+0E4E)
        '\u{0E40}'..='\u{0E4E}' => 5, // "V"
        // Thai digits (U+0E50-U+0E59)
        '\u{0E50}'..='\u{0E59}' => DIGIT_TYPE_ID, // "N"
        // Shared classes: "P" (punctuation), "A" (Latin), "N" (digits)
        _ => punct_latin_digit(c).unwrap_or(OTHER_TYPE_ID),
    }
}
```

English's `english_char_type_id` is a real, in-tree example of the same pattern applied to a Latin-script language: it adds "U" (uppercase), "W" (whitespace), and "Q" (apostrophe) as dedicated classes, and additionally *widens* the shared "P" class to cover ASCII punctuation (which the other languages leave as "O") -- a language's classification function is free to layer extra logic in front of `punct_latin_digit()`, not just append new classes after it.

### Design Tips for Character Types

- **Identify linguistically distinct categories** that correlate with word boundary patterns
- **Order matters** -- match arms are tried top to bottom, so put more specific classes before general ones
- **Consider high-frequency function words** as a separate type (as Chinese does with "F"), or, for a space-delimited language, whatever punctuation/case/diacritic distinctions actually correlate with boundaries (as English does with "U"/"W"/"Q")
- **Use extra logic inside an arm body** when a plain range is not enough (as Korean does with a codepoint test to split syllables with/without 받침)
- Reuse the shared `punct_latin_digit()` helper for the common "P"/"A"/"N" classes
- **Keep the code set prefix-free** -- no code may be a prefix of another (Korean's `SN`/`SF` work because `S` alone is not a code; a bare `"S"` is therefore rejected for **every** language by a unit test, not just Korean). The model loader decodes concatenated codes left to right when compiling packed feature keys, and prefix-freeness is what makes that decoding unambiguous
- **The type table needs at least 7 codes.** A shared test context (`packed_model.rs`'s `ctx_for`) asserts `codes.len() >= 7`; English's 7-code table is the current minimum. There is no fixed upper bound, but the dense feature tables (`BC`/`UC`/`TC`/`BQ`/`TQ`) scale roughly with `type_count^2` to `type_count^3`, so a much larger table trades model size and load time for classification granularity

## Step 4: Register the Type-Code Table and Classification Function

Add the language's ordered code table to `Language::type_codes()` (index = type id; shared codes first) and a dispatch arm in `Language::char_type_id()`. `char_type()` itself is derived from these two, so string codes and numeric ids cannot drift apart:

```rust
pub(crate) fn type_codes(self) -> &'static [&'static str] {
    match self {
        // ...
        Language::Thai => &["O", "P", "A", "N", "T", "V"],    // ← new
    }
}

pub(crate) fn char_type_id(self, c: char) -> u8 {
    match self {
        // ...
        Language::Thai => thai_char_type_id(c),    // ← new
    }
}
```

## Step 5: Decide on WC Feature Inclusion

The feature template is defined once in `packed_model.rs` (`TEMPLATES`), and `templates_for()` decides whether a language uses the trailing `WC1`--`WC4` char/type mixed templates:

```rust
pub(crate) fn templates_for(language: Language) -> &'static [Template] {
    match language {
        Language::Japanese | Language::Chinese => &TEMPLATES[..],
        Language::Korean | Language::English => &TEMPLATES[..BASE_TEMPLATE_COUNT], // 38 base templates
    }
}
```

This match is deliberately **exhaustive, with no wildcard arm** -- adding
`Thai` without adding it to one of the two arms is a compile error, not a
silent default. This is intentional: an earlier version of this match had
a `_ => &TEMPLATES[..BASE_TEMPLATE_COUNT]` fallback, which meant a new
language got the 38-template (no WC) configuration without anyone
deciding that on purpose. Make the WC decision explicitly, and back it
with a measurement rather than an assumption: train a tag-free model both
with and without WC on a held-out dev split and compare Word F1 (English's
comparison, documented in [English](english.md#no-wc-features), found WC
measured *worse*, not just unhelpful -- 98.68% without vs. 98.65% with).
As a starting heuristic: if your language's character types have enough
variety to make WC features informative, include them; if your type
system is low-entropy (like Korean's/English's dominant "SN"/"A" or
whitespace-dominated distribution), exclude them -- but verify with
numbers before committing bundled models to either choice.

## Step 6: Choose a Corpus Protocol

Litsea supports two corpus protocols, and the right choice depends on whether the language is written with spaces:

- **Space-separated** (the default): words joined with a single space, one sentence per line. Used for languages written without spaces (Japanese, Chinese) or where spacing carries no boundary signal.
- **Space-preserving TSV** (`--format tsv`, issue #152): tab-separated tokens where a token may itself be a literal space `" "`, so the original spacing survives into training as a first-class feature. Used for languages where the space itself is the strongest boundary signal (Korean, English).

If your language is written with spaces between words (like English, unlike Korean's inter-eojeol convention but the same underlying reasoning), use the space-preserving protocol:

```sh
conllu_file=$(bash scripts/download_udtreebank.sh -l en -o /tmp)
bash scripts/corpus_udtreebank.sh -s "$conllu_file" corpus.tsv
litsea extract -l english --format tsv --tag-free corpus.tsv features.txt
```

**If the source treebank has multiword tokens (contractions, clitics),
verify `corpus_udtreebank.sh -s` handles them correctly before trusting
the corpus.** UD CoNLL-U represents a contraction like English's `don't`
as a *range* line (e.g. `3-4  don't`) covering two word lines (`do`,
`n't`) that carry no space between them. `corpus_udtreebank.sh -s` treats
a range line specially: it emits no token of its own, suppresses space
insertion *between* the range's member words, and applies the range's own
`SpaceAfter` annotation only after the last member word. The invariant to
check for a new treebank is: **concatenating a range's member word forms
must reproduce the range's own surface form.** This held for every
multiword token in UD English-EWT (verified by scripting a comparison
across the full corpus before training); if it does not hold for your
treebank, the safe fallback is to emit the range's own form as a single
token instead of expanding its members. As a second, independent check,
reconstruct each sentence by concatenating its TSV tokens (space tokens
included) and diff the result against the CoNLL-U file's `# text =`
metadata line for that sentence -- this catches any spacing bug the
member-concatenation check alone would miss. When you change
`corpus_udtreebank.sh` itself, also regenerate an existing space-preserving
gold file (e.g. `resources/eval/korean_gsd_test.tsv`) and diff it against
the committed version -- it should come out byte-identical if your change
is additive.

## Step 7: Train the Bundled Segmentation Model

The bundled segmentation models are **not** trained with plain `litsea
train -t/-i` (AdaBoost boosting). They are trained as a 2-class Averaged
Perceptron and losslessly collapsed to the AdaBoost model format --
see [Pre-trained Models: Training
Procedure](../pre-trained-models.md#training-procedure) for the full
derivation and the exact five-step recipe
(extract → relabel `1`/`-1` to `B`/`O` → `train --perceptron` →
`scripts/collapse_binary_perceptron.py` → optionally prune). For a
space-preserving-protocol language:

```sh
litsea extract -l english --format tsv --tag-free corpus.tsv features.txt
sed -i 's/^1\t/B\t/; s/^-1\t/O\t/' features.txt
litsea train --perceptron --num-epochs 20 features.txt perceptron.model
scripts/collapse_binary_perceptron.py perceptron.model models/english.model
```

Do not guess the epoch count or the tag-free/WC decisions -- run a sweep on
a held-out **dev** split (never the test split, which should be touched
only once at the end) across a handful of epoch counts, and compare
tag-free vs. tagged and (per Step 5) WC vs. no-WC at the best epoch count
for each. English's sweep, for example, found quality peaking at 20
epochs and degrading slightly beyond it -- a one-shot low-epoch run would
have understated the model's real quality, and a one-shot high-epoch run
would have looked like overfitting was the ceiling rather than a
convergence point already passed.

## Step 8: Optionally Train a Two-Stage POS Model

If UPOS-tagged data is available for the language, you can additionally
train a [two-stage](../algorithm/two-stage-tagging.md) model:

```sh
bash scripts/corpus_udtreebank.sh -p "$conllu_file" pos_corpus.txt
litsea extract --pos -l english --stage2-features full pos_corpus.txt pos_features
litsea train --pos --num-epochs 50 pos_features models/english_pos.model
```

Sweep `--stage2-features` (`fast`/`balanced`/`full`) on a dev split the
same way as the epoch sweep above; the three bundled languages that use
this path each picked a different winner (see [Choosing a stage-2 feature
set](../algorithm/two-stage-tagging.md#choosing-a-stage-2-feature-set)).

**Known limitation for space-preserving-protocol languages**: the
two-stage extractor's corpus format has no space-preserving variant (`-p`
and `-s` are mutually exclusive), so `english_pos.model` is trained (and
evaluated) on the *unspaced* `word/POS` corpus, unlike `english.model`
itself. Whether this matters in practice depends on the language: Korean
stays close to its space-preserving numbers (99.90% unspaced vs. 99.91%
spaced) because its agglutinative morphology leaves strong boundary cues
even with spaces removed, while English's gap is large (70.33% unspaced
vs. 98.31% spaced) because English orthography carries almost no such
signal. Measure this gap for your language rather than assuming it will
be small, and document it prominently if it is not -- see
[English](english.md#english_posmodel) for the fully worked example of
how to explain a large gap like this to users.

## Step 9: Add Held-out Evaluation Gold Files

Generate held-out gold data from the treebank's **test** split (touched
only after your dev-split sweeps are done) and add it under
`resources/eval/`, following the existing naming convention
(`<language>_<treebank>_test.{txt,tsv}` for segmentation,
`<language>_<treebank>_test_pos.txt` for POS):

```sh
bash scripts/corpus_udtreebank.sh -s "$conllu_file_test" resources/eval/english_ewt_test.tsv
bash scripts/corpus_udtreebank.sh -p "$conllu_file_test" resources/eval/english_ewt_test_pos.txt
litsea evaluate -l english --format tsv models/english.model resources/eval/english_ewt_test.tsv
litsea evaluate -l english --pos models/english_pos.model resources/eval/english_ewt_test_pos.txt
```

Update `resources/eval/README.md` with the new file(s) and their
provenance/license (UD treebanks are typically CC BY-SA 4.0, distinct from
the rest of the repository's MIT/Apache-2.0 licensing). Record the
held-out numbers you get from `litsea evaluate` -- not numbers from
`litsea train`'s in-sample printout -- in the model's documentation
(Step 10 lists every doc page that needs the new numbers).

## Step 10: Add Tests and Documentation

This is the step most likely to be under-scoped: a new language touches
more test files than just `language.rs`/`segmenter.rs`. Work through this
checklist:

**Code tests:**

- `litsea/src/language.rs`: `ALL_LANGUAGES` (bump the array size), `test_language_from_str`, `test_parse_language_error_message` (new full error string), `test_language_display`, and a new `test_<language>_char_types` covering every type code plus a couple of shared-class and "O" cases
- `litsea/src/packed_model.rs`: `test_templates_for_language_gating` (assert the new language's template count), and the two hard-coded language-enumeration arrays in `test_pack_parse_roundtrip_unique_and_injective` and `test_dense_index_consistent_with_key_decode`
- `litsea/src/segmenter.rs`: a new `test_char_type_<language>` (mirroring the existing per-language ones)
- `litsea/src/word_features.rs`: add a case to the round-trip sample list (cheap, catches type-code encoding bugs early)
- `litsea-cli/src/main.rs`: the three `--language` help strings

**Model-dependent tests (need a trained model first):**

- `litsea/tests/golden.rs`: a new `golden_segment_<language>` (and, if you trained a POS model, `golden_segment_with_pos_<language>_two_stage`). **Pin the model's actual output, not an idealized one** -- write the test with a `println!` of the real output first, copy that into the assertion, then delete the debug print. Immediately after, **sabotage-check it**: temporarily change one expected value to something wrong, confirm the test goes RED, then restore the correct value. A golden test that never failed during its own creation has not proven it protects anything
- `litsea/src/segmenter.rs`: a differential test (`test_segment_differential_<language>_model`) comparing the packed scorer against the string-keyed reference, and, if the model is tag-free, a `segment_into` tiling/parity case
- `litsea/benches/bench.rs`: add the new language to all four per-language tuple lists (`bench_segment_short`, `bench_external_corpus`'s two case lists, `bench_segment_into`), which requires a corpus file under `resources/` (a public-domain text of a similar size to the existing per-language corpora)
- `litsea-cli/tests/cli.rs`: at least one segmentation smoke test pinning CLI output end-to-end

**Verification commands** (run all of these before considering the language done):

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench -- external_corpus   # sanity-check the new bench cases run; decide if pruning is needed
markdownlint-cli2 "docs/src/**/*.md"
markdownlint-cli2 "docs/ja/src/**/*.md"
mdbook build docs
mdbook build docs/ja
```

**Documentation** (English source first, then the Japanese mirror under `docs/ja/src/`, per this project's documentation policy):

- A new `language-support/<language>.md` page (use [English](english.md) or [Korean](korean.md) as the template) plus a `docs/src/SUMMARY.md` entry (and its `docs/ja/src/SUMMARY.md` mirror)
- `language-support/overview.md` and `algorithm/character-type-classification.md` -- both have a per-language table/section to extend
- `pre-trained-models.md` -- a model card per bundled model, plus the "Tag-Free (Pointwise) Models" and "Two-Stage POS Tagging Models" comparison tables if applicable
- Root `README.md` and the crate-level docs in `litsea/src/lib.rs`
- Before considering the sweep complete, run `grep -rln "<an existing language's name>" docs/src docs/ja/src` and check every file it finds -- language-enumerating docs are easy to miss (this project's own experience: issue #165 skipped this check once and left 14 stale mentions for a later PR to clean up)
