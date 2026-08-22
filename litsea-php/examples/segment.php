<?php

declare(strict_types=1);

/**
 * Segment a sentence, with and without POS tags.
 *
 * Usage:
 *   php -d extension=../target/release/liblitsea_php.so \
 *       examples/segment.php ../models/japanese.model "これはテストです。"
 */

if ($argc !== 3) {
    fwrite(STDERR, "usage: php examples/segment.php <model> <text>\n");
    exit(2);
}

[, $modelPath, $text] = $argv;

// The model file identifies its own kind, so nothing here declares whether
// this is a POS model - hasPos() reports what was loaded.
$segmenter = Litsea\Segmenter::open('japanese', $modelPath);
printf("model: %s (hasPos=%s)\n", $modelPath, $segmenter->hasPos() ? 'true' : 'false');
printf("tokens: %s\n", implode(' ', $segmenter->segment($text)));

if ($segmenter->hasPos()) {
    echo "tagged:\n";
    foreach ($segmenter->segmentWithPos($text) as $token) {
        printf("  %s\t%s\t[%d:%d]\n", $token->surface, $token->pos, $token->start, $token->end);
    }
}
