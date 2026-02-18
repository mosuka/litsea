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
| `segment_japanese_short` | Segment a short Japanese sentence |
| `segment_japanese_long` | Segment the full Bocchan novel (~300 KB) |
| `segment_chinese_short` | Segment a short Chinese sentence |
| `segment_korean_short` | Segment a short Korean sentence |
| `get_type_hiragana` | Character type classification |
| `add_corpus` | Corpus ingestion for training |
| `char_type_patterns_japanese` | Pattern compilation cost |
| `predict` | Single AdaBoost prediction |

## HTML Reports

Criterion generates detailed HTML reports with statistics and comparison graphs at:

```text
target/criterion/report/index.html
```

Open this file in a browser after running benchmarks to view:

- Iteration times with confidence intervals
- Throughput measurements
- Comparison with previous runs (automatic regression detection)

## Interpreting Results

Key performance factors:

- **Segmentation** is linear in input length (O(n))
- **Pattern compilation** (regex) is the most expensive one-time cost -- `Segmenter::new()` caches patterns
- **Prediction** at each position depends on the number of features (38-42, constant)
- **Model loading** time is proportional to the model file size
