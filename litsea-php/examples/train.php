<?php

declare(strict_types=1);

/**
 * Train a segmentation model.
 *
 * Usage:
 *   php -d extension=../target/release/liblitsea_php.so \
 *       examples/train.php corpus.txt out.model
 *
 * The corpus is one sentence per line, with words separated by spaces:
 *
 *   これ は テスト です 。
 *
 * Run this from the CLI SAPI: training blocks the process, which is not
 * something a web request should do.
 */

if ($argc !== 3) {
    fwrite(STDERR, "usage: php examples/train.php <corpus> <model>\n");
    exit(2);
}

[, $corpus, $model] = $argv;

$workDir = sys_get_temp_dir() . '/litsea-' . bin2hex(random_bytes(6));
mkdir($workDir, 0o700, true);
$features = $workDir . '/features.txt';

printf("extracting features from %s ...\n", $corpus);
(new Litsea\Extractor('japanese'))->extract($corpus, $features);

echo "training ...\n";
$metrics = (new Litsea\Trainer(0.01, 10000, $features))->train($model);

printf("wrote %s\n", $model);
printf("  accuracy:  %.2f%%\n", $metrics->accuracy);
printf("  precision: %.2f%%\n", $metrics->precision);
printf("  recall:    %.2f%%\n", $metrics->recall);
printf("  instances: %d\n", $metrics->numInstances);
