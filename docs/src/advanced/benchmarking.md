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
| `segment_short/adaboost/{ja,zh,ko}` | Segment a short sentence (AdaBoost) |
| `segment_short/averaged_perceptron/{ja,zh,ko}` | Segment + POS tag a short sentence |
| `segment_long_japanese/{adaboost,averaged_perceptron}` | Process the full Bocchan novel (~300 KB) |
| `get_type_hiragana` | Character type classification |
| `add_corpus` | Corpus ingestion for training |
| `predict_adaboost` | Single AdaBoost prediction |

Models are loaded synchronously with `load_model_from_path` — no async runtime is involved in the benchmarks.

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
- **Character classification** is a direct `match` on character ranges (a few nanoseconds; no setup cost)
- **Prediction** at each position depends on the number of features (38-42, constant)
- **Model loading** time is proportional to the model file size
