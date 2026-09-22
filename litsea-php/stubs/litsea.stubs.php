<?php

/**
 * Stubs for the litsea PHP extension (Packagist: mosuka/litsea).
 *
 * Generated from the compiled extension by litsea-php/tools/generate-stubs.php;
 * do not edit by hand. Point PHPStan (stubFiles) or Psalm (stubs) at this file.
 * Never include it at runtime next to the real extension: the classes would be
 * declared twice.
 *
 * @generated
 */

namespace Litsea;

function supported_languages(): array {}

function version(): string {}

class BinaryMetrics
{
    /** @var float */
    public $accuracy;

    /** @var int */
    public $falseNegatives;

    /** @var int */
    public $falsePositives;

    /** @var int */
    public $numInstances;

    /** @var float */
    public $precision;

    /** @var float */
    public $recall;

    /** @var int */
    public $trueNegatives;

    /** @var int */
    public $truePositives;

    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class CancelToken
{
    public function __construct() {}

    public function cancel(): void {}

    public function isCancelled(): bool {}

    public function reset(): void {}
}

class Extractor
{
    public function __construct(string $language) {}

    public function extract(string $corpus_path, string $features_path, bool $tsv = false, bool $tag_free = false): void {}

    public function extractTwoStage(string $corpus_path, string $output_prefix, ?string $feature_set = null, bool $tsv = false): void {}
}

class MulticlassMetrics
{
    /** @var float */
    public $accuracy;

    /** @var float */
    public $macroPrecision;

    /** @var float */
    public $macroRecall;

    /** @var int */
    public $numInstances;

    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class PerceptronTrainer
{
    public function __construct(int $num_epochs, string $features_path) {}

    public function train(string $model_path, ?CancelToken $cancel = null): MulticlassMetrics {}
}

class Segmenter
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}

    public static function fromBytes(string $language, string $data): Segmenter {}

    public static function fromUri(string $language, string $uri): Segmenter {}

    public function hasPos(): bool {}

    public function language(): string {}

    public static function open(string $language, string $path): Segmenter {}

    public function segment(string $text): array {}

    public function segmentBatch(array $texts): array {}

    public function segmentTokens(string $text): array {}

    public function segmentWithPos(string $text): array {}

    public function segmentWithPosBatch(array $texts): array {}
}

class Token implements \Stringable
{
    /** @var int */
    public $end;

    /** @var string|null */
    public $pos;

    /** @var int */
    public $start;

    /** @var string */
    public $surface;

    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}

    public function __toString(): string {}
}

class Trainer
{
    public function __construct(float $threshold, int $num_iterations, string $features_path) {}

    public function loadModel(string $model_uri): void {}

    public function train(string $model_path, ?CancelToken $cancel = null): BinaryMetrics {}
}

class TwoStageMetrics
{
    /** @var float */
    public $stage1Accuracy;

    /** @var int */
    public $stage1NumInstances;

    /** @var float */
    public $stage2Accuracy;

    /** @var int */
    public $stage2NumInstances;

    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class TwoStageTrainer
{
    public function __construct(int $num_epochs, string $features_prefix, float $dominance = 0.99) {}

    public function isAvailable(): bool {}

    public function train(string $model_path, ?CancelToken $cancel = null): TwoStageMetrics {}
}

class LitseaException extends \Exception
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class InvalidArgumentException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class IoException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class ModelException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class ParseException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class PosUnavailableException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}

class UnsupportedException extends LitseaException
{
    /** Not constructible from PHP: instances come from the extension. */
    private function __construct() {}
}
