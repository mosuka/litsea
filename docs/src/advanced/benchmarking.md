# Benchmarking

Litsea includes a Criterion benchmark suite for measuring performance.

## Running Benchmarks

```sh
cargo bench --bench bench
```

Or via the Makefile:

```sh
make bench
```

## Benchmark Suite

The benchmarks are defined in `litsea/benches/bench.rs`:

| Benchmark | Description |
|-----------|------------|
| `segment_short/adaboost/{japanese,chinese,korean}` | Segment a short sentence (AdaBoost) |
| `segment_short/averaged_perceptron/{japanese,chinese,korean}` | Segment + POS tag a short sentence |
| `segment_long_japanese/{adaboost,averaged_perceptron}` | Process the full Bocchan novel (~300 KB) |
| `external_corpus/*` | Corpus throughput, mirroring tokenizer-speed-bench (see below) |
| `char_type_hiragana` | Character type classification |
| `add_corpus` | Corpus ingestion for training |
| `predict_adaboost` | Single AdaBoost prediction |

Models are loaded synchronously with `load_model_from_path` — no async runtime is involved in the benchmarks.

## Corpus Throughput (`external_corpus`)

The `external_corpus` group reproduces the seven litsea benches of the
external [tokenizer-speed-bench](https://github.com/mosuka/tokenizer-speed-bench)
harness in-repo, so throughput regressions can be caught with `cargo bench`
alone:

```sh
cargo bench --bench bench -- external_corpus
```

| Bench id | Model | Corpus |
|----------|-------|--------|
| `japanese` | japanese.model | wagahaiwa_nekodearu.txt |
| `japanese-rwcp` | RWCP.model | wagahaiwa_nekodearu.txt |
| `japanese-two-stage` | japanese_pos.model | wagahaiwa_nekodearu.txt |
| `korean` | korean.model | mujeong.txt |
| `korean-two-stage` | korean_pos.model | mujeong.txt |
| `chinese` | chinese.model | rulin_waishi.txt |
| `chinese-two-stage` | chinese_pos.model | rulin_waishi.txt |

The `*-two-stage` benches were added alongside the [two-stage
architecture](../algorithm/two-stage-tagging.md) (#147/#169); they are not
part of the original seven tokenizer-speed-bench-mirroring benches above.

One iteration segments every line of the corpus (unfiltered, like the
external harness), and the group sets `Throughput::Elements` to the
corpus's newline-free character count, so Criterion's `elem/s` figures
read directly as **chars/sec**.

The corpora live in `resources/`, byte-identical to the external harness:

| Corpus | Size | Source |
|--------|------|--------|
| wagahaiwa_nekodearu.txt | ~1.1 MB | 吾輩は猫である (Natsume Soseki), Aozora Bunko, public domain |
| mujeong.txt | ~786 KB | 무정 (Yi Kwang-su, 1917), ko.wikisource, public domain — naturally spaced modern Korean, matching the space-aware korean.model |
| rulin_waishi.txt | ~985 KB | 儒林外史 (Wu Jingzi), zh.wikisource, public domain — Traditional Chinese, matching UD Chinese-GSD |

Numbers are comparable to, but not identical with, the published
tokenizer-speed-bench figures, for two methodological reasons: Criterion
uses in-process warmup and sampling instead of 101 process-interleaved
single passes, and `cargo bench` inherits litsea's tuned release profile
(thin LTO, single codegen unit) while the external bench crates build with
the default release profile.

## API Comparison (`segment_into`)

The `segment_into` group pairs the owned-output `segment()` API against
the buffer-reusing `segment_into()` API (issue #184) on the same three
segmentation corpora as `external_corpus` (same per-line workload,
chars/sec via `Throughput::Elements`):

| Bench ID | API |
|----------|-----|
| `japanese-strings` / `korean-strings` / `chinese-strings` | `segment()` (one `String` per token, fresh scratch per call) |
| `japanese-ranges` / `korean-ranges` / `chinese-ranges` | `segment_into()` with one reused `SegmentBuffer` |

Compare the two ids of a language within one run: their difference is the
per-call allocation cost the buffer-reusing API removes. The scoring work
is identical (`segment()` is a wrapper over `segment_into()`).

```sh
cargo bench -- segment_into
```

### Engine vs. CLI Numbers

Everything in this chapter measures **single-threaded engine throughput**.
The CLI's `segment --threads N` (issue #185) additionally scales
wall-clock batch time across cores at the process level; those two kinds
of numbers are not comparable — a `--threads 8` wall-clock figure is not
an engine speedup, and engine chars/sec figures say nothing about CLI
thread scaling. When reporting CLI-level scaling, state the thread count
and measure with the same paired discipline described below.

### Run-to-Run Variance

The published figures in this book (including the throughput figures on
the [Two-Stage Tagging](../algorithm/two-stage-tagging.md) and
[Pre-trained Models](../pre-trained-models.md) pages) are measured on this
project's development machine, not dedicated, idle benchmarking
hardware. Three
consecutive `external_corpus` runs of the same build showed spreads of
10-20% on individual bench ids -- large enough that a single run should
not be read as a precise figure. Where a page reports a range or an
explicit "N runs" note, that reflects this variance directly; where a
single number is given, treat it as accurate to roughly this same range.
Comparing two models measured *in the same run* (rather than against a
previously published number from a different run) cancels out most of
this variance, since both models see the same machine state.

## HTML Reports

Criterion generates detailed HTML reports with statistics and comparison graphs at:

```text
target/criterion/report/index.html
```

Open this file in a browser after running benchmarks to view:

- Iteration times with confidence intervals
- Throughput measurements
- Comparison with previous runs (automatic regression detection)

## Release Profile

`cargo bench` inherits the release profile, which enables thin LTO and a
single codegen unit (see the workspace `Cargo.toml`). Benchmark numbers
therefore reflect the optimized configuration that release binaries ship
with; a plain `cargo build` (dev profile) is significantly slower and not
representative.

## Interpreting Results

Key performance factors:

- **Segmentation** is linear in input length (O(n))
- **Character classification** is a direct `match` on character ranges (a few nanoseconds; no setup cost)
- **Prediction** at each position depends on the number of features (38-42, constant)
- **Model loading** time is proportional to the model file size
