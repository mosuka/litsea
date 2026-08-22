<?php

declare(strict_types=1);

namespace Litsea\Tests;

use Litsea\CancelToken;
use Litsea\Extractor;
use Litsea\InvalidArgumentException;
use Litsea\IoException;
use Litsea\PerceptronTrainer;
use Litsea\Segmenter;
use Litsea\Trainer;
use Litsea\TwoStageTrainer;

/**
 * Feature extraction, training, and cancellation.
 */
final class TrainingTest extends LitseaTestCase
{
    private const SENTENCES = [
        'これ は テスト です 。',
        '隣 の 客 は よく 柿 食う 客 だ',
        '東京 都 から 神奈川 県 へ 引っ越し た',
    ];

    private const POS_SENTENCES = [
        'これ/PRON は/ADP テスト/NOUN です/AUX 。/PUNCT',
        '隣/NOUN の/ADP 客/NOUN は/ADP よく/ADV 柿/NOUN 食う/VERB 客/NOUN だ/AUX',
        '東京/PROPN 都/NOUN から/ADP 神奈川/PROPN 県/NOUN へ/ADP 引っ越し/VERB た/AUX',
    ];

    /**
     * Writes a small corpus, repeated so training has something to learn.
     *
     * @param string[] $sentences
     */
    private static function writeCorpus(string $path, array $sentences, int $repeats = 20): string
    {
        $lines = [];
        for ($i = 0; $i < $repeats; $i++) {
            foreach ($sentences as $sentence) {
                $lines[] = $sentence;
            }
        }
        file_put_contents($path, implode("\n", $lines) . "\n");

        return $path;
    }

    public function testExtractThenTrainRoundTrip(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus.txt', self::SENTENCES);
        $features = $dir . '/features.txt';
        $model = $dir . '/trained.model';

        (new Extractor('japanese'))->extract($corpus, $features);
        $this->assertGreaterThan(0, filesize($features));

        $metrics = (new Trainer(0.01, 20, $features))->train($model);
        $this->assertGreaterThan(0, $metrics->numInstances);
        $this->assertGreaterThanOrEqual(0.0, $metrics->accuracy);
        $this->assertLessThanOrEqual(100.0, $metrics->accuracy);

        // The CLI is the independent check that the file is a valid model.
        [$line] = self::runCli(['segment', '-l', 'japanese', $model], "これはテストです。\n");
        $this->assertNotSame('', $line);
        $this->assertSame(
            $line,
            implode(' ', Segmenter::open('japanese', $model)->segment('これはテストです。'))
        );
    }

    public function testTagFreeExtractionIsSmaller(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus.txt', self::SENTENCES);
        $extractor = new Extractor('japanese');

        $extractor->extract($corpus, $dir . '/full.txt');
        $extractor->extract($corpus, $dir . '/lean.txt', false, true);

        $this->assertLessThan(filesize($dir . '/full.txt'), filesize($dir . '/lean.txt'));
    }

    public function testTwoStageTrainingRoundTrip(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus_pos.txt', self::POS_SENTENCES);
        $prefix = $dir . '/features';
        $model = $dir . '/two_stage.model';

        (new Extractor('japanese'))->extractTwoStage($corpus, $prefix, 'fast');
        foreach (['stage1', 'stage2', 'lexicon'] as $suffix) {
            $this->assertFileExists($prefix . '.' . $suffix);
        }

        $trainer = new TwoStageTrainer(3, $prefix);
        $this->assertTrue($trainer->isAvailable());
        $metrics = $trainer->train($model);
        $this->assertGreaterThan(0, $metrics->stage1NumInstances);
        $this->assertGreaterThan(0, $metrics->stage2NumInstances);

        $seg = Segmenter::open('japanese', $model);
        $this->assertTrue($seg->hasPos());
        $tokens = $seg->segmentWithPos('これはテストです。');
        $this->assertNotEmpty($tokens);
        foreach ($tokens as $token) {
            $this->assertNotNull($token->pos);
        }
    }

    public function testTwoStageTrainerCannotBeReused(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus_pos.txt', self::POS_SENTENCES);
        $prefix = $dir . '/features';
        $model = $dir . '/two_stage.model';

        (new Extractor('japanese'))->extractTwoStage($corpus, $prefix);
        $trainer = new TwoStageTrainer(1, $prefix);
        $trainer->train($model);
        $this->assertFalse($trainer->isAvailable());

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/already been used/');
        $trainer->train($model);
    }

    public function testPerceptronTrainer(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus_pos.txt', self::POS_SENTENCES);
        $prefix = $dir . '/features';

        (new Extractor('japanese'))->extractTwoStage($corpus, $prefix);
        $metrics = (new PerceptronTrainer(2, $prefix . '.stage2'))->train($dir . '/perceptron.model');

        $this->assertGreaterThan(0, $metrics->numInstances);
        $this->assertFileExists($dir . '/perceptron.model');
    }

    public function testCancelBeforeTrainingStillWritesAModel(): void
    {
        // PHP cannot cancel a run in flight -- a request is single-threaded --
        // so the token has to be cancelled before `train()`. Cancelling still
        // means "stop early and keep the partial model", not "fail".
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus.txt', self::SENTENCES);
        $features = $dir . '/features.txt';
        $model = $dir . '/cancelled.model';

        (new Extractor('japanese'))->extract($corpus, $features);

        $cancel = new CancelToken();
        $cancel->cancel();
        $this->assertTrue($cancel->isCancelled());

        $metrics = (new Trainer(0.01, 100000, $features))->train($model, $cancel);
        $this->assertGreaterThan(0, $metrics->numInstances);
        $this->assertFileExists($model);
    }

    public function testCancelTokenReset(): void
    {
        $token = new CancelToken();
        $this->assertFalse($token->isCancelled());
        $token->cancel();
        $this->assertTrue($token->isCancelled());
        $token->reset();
        $this->assertFalse($token->isCancelled());
    }

    public function testUnknownFeatureSetThrows(): void
    {
        $dir = self::tempDir();
        $corpus = self::writeCorpus($dir . '/corpus_pos.txt', self::POS_SENTENCES, 1);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/turbo/');
        (new Extractor('japanese'))->extractTwoStage($corpus, $dir . '/features', 'turbo');
    }

    public function testMissingFeaturesFileThrows(): void
    {
        $this->expectException(IoException::class);
        new Trainer(0.01, 10, self::tempDir() . '/missing.txt');
    }
}
