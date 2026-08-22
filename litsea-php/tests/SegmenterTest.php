<?php

declare(strict_types=1);

namespace Litsea\Tests;

use Litsea\InvalidArgumentException;
use Litsea\IoException;
use Litsea\LitseaException;
use Litsea\ModelException;
use Litsea\ParseException;
use Litsea\PosUnavailableException;
use Litsea\Segmenter;
use Litsea\Token;
use PHPUnit\Framework\Attributes\DataProvider;

/**
 * Segmentation, POS tagging, and the exception hierarchy.
 */
final class SegmenterTest extends LitseaTestCase
{
    /** @return array<string, array{string, string, string}> */
    public static function segmentationCases(): array
    {
        return [
            'japanese' => ['japanese', 'japanese.model', 'これはテストです。'],
            'chinese' => ['chinese', 'chinese.model', '我喜欢吃中国菜。'],
            'korean' => ['korean', 'korean.model', '안녕하세요 반갑습니다'],
            'english' => ['english', 'english.model', 'The quick brown fox jumps over the lazy dog.'],
        ];
    }

    /** @return array<string, array{string, string, string}> */
    public static function posCases(): array
    {
        return [
            'japanese' => ['japanese', 'japanese_pos.model', 'これはテストです。'],
            'korean' => ['korean', 'korean_pos.model', '안녕하세요 반갑습니다'],
        ];
    }

    #[DataProvider('segmentationCases')]
    public function testSegmentMatchesTheCli(string $language, string $model, string $sentence): void
    {
        // Compare the rendered line rather than a re-split of it: the CLI joins
        // tokens with a space, so for Korean and English -- where whitespace is
        // its own token -- splitting the output again cannot recover them.
        [$expected] = self::runCli(['segment', '-l', $language, self::modelPath($model)], $sentence . "\n");

        $seg = Segmenter::open($language, self::modelPath($model));
        $this->assertSame($expected, implode(' ', $seg->segment($sentence)));
    }

    #[DataProvider('posCases')]
    public function testSegmentWithPosMatchesTheCli(string $language, string $model, string $sentence): void
    {
        [$expected] = self::runCli(
            ['segment', '-l', $language, '--pos', self::modelPath($model)],
            $sentence . "\n"
        );

        $seg = Segmenter::open($language, self::modelPath($model));
        $rendered = implode(' ', array_map(
            static fn (Token $token) => $token->surface . '/' . $token->pos,
            $seg->segmentWithPos($sentence)
        ));
        $this->assertSame($expected, $rendered);
    }

    #[DataProvider('segmentationCases')]
    public function testByteOffsetsReconstructTheInput(string $language, string $model, string $sentence): void
    {
        $seg = Segmenter::open($language, self::modelPath($model));
        $tokens = $seg->segmentTokens($sentence);

        $this->assertNotEmpty($tokens);
        $expectedStart = 0;
        $joined = '';
        foreach ($tokens as $token) {
            $this->assertSame($expectedStart, $token->start, 'tokens must tile the input');
            // PHP strings are byte strings, so substr() works with byte offsets.
            $this->assertSame($token->surface, substr($sentence, $token->start, $token->end - $token->start));
            $this->assertNull($token->pos);
            $expectedStart = $token->end;
            $joined .= $token->surface;
        }
        $this->assertSame(strlen($sentence), $expectedStart);
        $this->assertSame($sentence, $joined);
    }

    public function testWhitespaceIsItsOwnToken(): void
    {
        $seg = Segmenter::open('korean', self::modelPath('korean.model'));
        $this->assertSame(['안녕하세요', ' ', '반갑습니다'], $seg->segment('안녕하세요 반갑습니다'));
    }

    public function testSegmentBatchMatchesSingleCalls(): void
    {
        $seg = Segmenter::open('japanese', self::modelPath('japanese.model'));
        $sentences = ['これはテストです。', '', '東京都から神奈川県へ引っ越した'];

        $batched = $seg->segmentBatch($sentences);
        $this->assertSame(array_map(static fn ($s) => $seg->segment($s), $sentences), $batched);
        $this->assertSame([], $batched[1]);
    }

    public function testSegmentWithPosBatchMatchesSingleCalls(): void
    {
        $seg = Segmenter::open('japanese', self::modelPath('japanese_pos.model'));
        $sentences = ['これはテストです。', '東京都から神奈川県へ引っ越した'];

        $batched = $seg->segmentWithPosBatch($sentences);
        $this->assertCount(2, $batched);
        foreach ($sentences as $index => $sentence) {
            $expected = array_map(
                static fn (Token $token) => $token->surface . '/' . $token->pos,
                $seg->segmentWithPos($sentence)
            );
            $actual = array_map(
                static fn (Token $token) => $token->surface . '/' . $token->pos,
                $batched[$index]
            );
            $this->assertSame($expected, $actual);
        }
    }

    public function testModelKindIsDetected(): void
    {
        $this->assertFalse(Segmenter::open('ja', self::modelPath('japanese.model'))->hasPos());
        $this->assertTrue(Segmenter::open('ja', self::modelPath('japanese_pos.model'))->hasPos());
    }

    public function testLanguageNamesAndCodesAreInterchangeable(): void
    {
        $expected = Segmenter::open('japanese', self::modelPath('japanese.model'))->segment('これはテストです。');
        foreach (['ja', 'JA', 'japanese', 'Japanese'] as $name) {
            $this->assertSame($expected, Segmenter::open($name, self::modelPath('japanese.model'))->segment('これはテストです。'));
        }
    }

    public function testLoadingSourcesAgree(): void
    {
        $path = self::modelPath('japanese.model');
        $sentence = 'これはテストです。';

        $fromPath = Segmenter::open('japanese', $path);
        $fromBytes = Segmenter::fromBytes('japanese', file_get_contents($path));
        $fromUri = Segmenter::fromUri('japanese', $path);

        $this->assertSame($fromPath->segment($sentence), $fromBytes->segment($sentence));
        $this->assertSame($fromPath->segment($sentence), $fromUri->segment($sentence));
    }

    public function testLanguageAccessor(): void
    {
        $seg = Segmenter::open('korean', self::modelPath('korean.model'));
        $this->assertSame('korean', $seg->language());
    }

    public function testPosOnSegmentationModelThrows(): void
    {
        $seg = Segmenter::open('japanese', self::modelPath('japanese.model'));

        $this->expectException(PosUnavailableException::class);
        $this->expectExceptionMessageMatches('/two-stage POS model/');
        $seg->segmentWithPos('これはテストです。');
    }

    public function testUnknownLanguageThrows(): void
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/klingon/');
        Segmenter::open('klingon', self::modelPath('japanese.model'));
    }

    public function testMissingModelThrows(): void
    {
        $this->expectException(IoException::class);
        Segmenter::open('japanese', self::modelPath('does-not-exist.model'));
    }

    public function testMalformedModelThrows(): void
    {
        $dir = self::tempDir();
        file_put_contents($dir . '/broken.model', "this is not a model\n");

        $this->expectException(ParseException::class);
        Segmenter::open('japanese', $dir . '/broken.model');
    }

    public function testLegacyJointModelThrows(): void
    {
        $dir = self::tempDir();
        // A bare integer first line is the joint class-count header.
        file_put_contents($dir . '/joint.model', "17\nfoo\t1.0\n");

        $this->expectException(ModelException::class);
        $this->expectExceptionMessageMatches('/no longer supported/');
        Segmenter::open('japanese', $dir . '/joint.model');
    }

    public function testEveryExceptionDerivesFromTheBase(): void
    {
        foreach ([
            InvalidArgumentException::class,
            IoException::class,
            ModelException::class,
            ParseException::class,
            PosUnavailableException::class,
        ] as $class) {
            $this->assertTrue(is_subclass_of($class, LitseaException::class), $class);
        }

        // One catch is enough for anything the binding throws.
        $this->expectException(LitseaException::class);
        Segmenter::open('klingon', self::modelPath('japanese.model'));
    }
}
