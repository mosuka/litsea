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

    /**
     * User dictionary cases (#263): the model splits each registered word
     * without the dictionary, and the English word contains a space.
     *
     * @return array<string, array{string, string, list<string>, string}>
     */
    public static function userWordCases(): array
    {
        return [
            'japanese snowman' => ['japanese', 'japanese.model', ['雪だるま', '新幹線'], '雪だるまを作る'],
            'japanese shinkansen' => ['japanese', 'japanese.model', ['雪だるま', '新幹線'], '新幹線で帰る'],
            'english space' => ['english', 'english.model', ['New York'], 'I live in New York.'],
        ];
    }

    /**
     * Writes a `--user-words` file for the CLI.
     *
     * @param list<string> $words
     */
    private static function writeWords(array $words): string
    {
        $path = self::tempDir() . '/words.txt';
        file_put_contents($path, implode('', array_map(static fn (string $word) => $word . "\n", $words)));

        return $path;
    }

    /** @param list<string> $words */
    #[DataProvider('userWordCases')]
    public function testUserWordsMatchTheCli(string $language, string $model, array $words, string $sentence): void
    {
        [$expected] = self::runCli(
            ['segment', '-l', $language, '--user-words', self::writeWords($words), self::modelPath($model)],
            $sentence . "\n"
        );

        $seg = Segmenter::open($language, self::modelPath($model), $words);
        $this->assertSame($expected, implode(' ', $seg->segment($sentence)));
        // The dictionary must actually matter for these cases.
        $this->assertNotSame(Segmenter::open($language, self::modelPath($model))->segment($sentence), $seg->segment($sentence));
    }

    public function testUserWordsWithPosMatchTheCli(): void
    {
        $sentence = '雪だるまを作る';
        $words = ['雪だるま'];
        [$expected] = self::runCli(
            ['segment', '-l', 'japanese', '--pos', '--user-words', self::writeWords($words), self::modelPath('japanese_pos.model')],
            $sentence . "\n"
        );

        $seg = Segmenter::open('japanese', self::modelPath('japanese_pos.model'), $words);
        $tokens = $seg->segmentWithPos($sentence);
        $rendered = implode(' ', array_map(static fn (Token $token) => $token->surface . '/' . $token->pos, $tokens));
        $this->assertSame($expected, $rendered);
        $this->assertSame('雪だるま', $tokens[0]->surface);
    }

    public function testUserWordsOnEveryConstructor(): void
    {
        $path = self::modelPath('japanese.model');
        $sentence = '雪だるまを作る';
        $expected = ['雪だるま', 'を', '作る'];

        $this->assertSame($expected, Segmenter::open('japanese', $path, ['雪だるま'])->segment($sentence));
        $this->assertSame($expected, Segmenter::fromBytes('japanese', file_get_contents($path), ['雪だるま'])->segment($sentence));
        $this->assertSame($expected, Segmenter::fromUri('japanese', $path, ['雪だるま'])->segment($sentence));
        // Named arguments work too.
        $this->assertSame($expected, Segmenter::open('japanese', $path, user_words: ['雪だるま'])->segment($sentence));
    }

    public function testUserWordsGate(): void
    {
        $path = self::modelPath('japanese.model');
        $sentence = '雪だるまを作る';
        $plain = Segmenter::open('japanese', $path)->segment($sentence);

        // Gate 0 applies a word only where the model already agrees with it.
        $this->assertSame($plain, Segmenter::open('japanese', $path, ['雪だるま'], 0.0)->segment($sentence));
        $this->assertSame(['雪だるま', 'を', '作る'], Segmenter::open('japanese', $path, ['雪だるま'], 1.0)->segment($sentence));
        // An empty array or null is no dictionary at all.
        $this->assertSame($plain, Segmenter::open('japanese', $path, [])->segment($sentence));
        $this->assertSame($plain, Segmenter::open('japanese', $path, null, null)->segment($sentence));
    }

    public function testOutOfRangeUserWordsGateThrows(): void
    {
        foreach ([['雪だるま'], null] as $words) {
            try {
                Segmenter::open('japanese', self::modelPath('japanese.model'), $words, 1.5);
                $this->fail('expected InvalidArgumentException');
            } catch (InvalidArgumentException $error) {
                $this->assertMatchesRegularExpression('/0\.0\.\.=1\.0/', $error->getMessage());
            }
        }
    }

    /**
     * Tagged user dictionary entries (#264): the tagger gives these words NOUN
     * without a tag (`ポケモン` through the lexicon, `チャットGPT` through the
     * full argmax). `楓` is a one-character, tag-only entry (#276).
     */
    private const USER_WORD_TAGS = ['チャットGPT' => 'PROPN', 'ポケモン' => 'PROPN', '楓' => 'PROPN'];

    private const TAGGED_SENTENCES = ['チャットGPTに聞く', 'ポケモンを集める', '雪だるまを作る', '楓が笑った。'];

    /** Renders `segmentWithPos()` the way `litsea segment --pos` prints it. */
    private static function tagged(Segmenter $segmenter, string $sentence): string
    {
        return implode(' ', array_map(
            static fn (Token $token) => $token->surface . '/' . $token->pos,
            $segmenter->segmentWithPos($sentence)
        ));
    }

    public function testUserWordTagsMatchTheCli(): void
    {
        $model = self::modelPath('japanese_pos.model');
        $lines = [];
        foreach (self::USER_WORD_TAGS as $word => $tag) {
            $lines[] = $word . "\t" . $tag;
        }
        $lines[] = '雪だるま';
        $expected = self::runCli(
            ['segment', '-l', 'japanese', '--pos', '--user-words', self::writeWords($lines), $model],
            implode('', array_map(static fn (string $sentence) => $sentence . "\n", self::TAGGED_SENTENCES))
        );

        $seg = Segmenter::open('japanese', $model, ['雪だるま'], null, self::USER_WORD_TAGS);
        $actual = array_map(static fn (string $sentence) => self::tagged($seg, $sentence), self::TAGGED_SENTENCES);
        $this->assertSame($expected, $actual);
        $this->assertStringStartsWith('チャットGPT/PROPN ', self::tagged($seg, 'チャットGPTに聞く'));
        // The tags must actually matter: untagged, the same words come out as NOUN.
        $untagged = Segmenter::open('japanese', $model, ['雪だるま', ...array_keys(self::USER_WORD_TAGS)]);
        $this->assertStringStartsWith('チャットGPT/NOUN ', self::tagged($untagged, 'チャットGPTに聞く'));
        $this->assertNotSame(self::tagged($untagged, 'ポケモンを集める'), self::tagged($seg, 'ポケモンを集める'));
        $this->assertStringStartsWith('楓/PROPN ', self::tagged($seg, '楓が笑った。'));
        $this->assertStringStartsWith('楓/NOUN ', self::tagged($untagged, '楓が笑った。'));
    }

    public function testUserWordTagsOnEveryConstructor(): void
    {
        $path = self::modelPath('japanese_pos.model');
        $sentence = 'チャットGPTに聞く';

        $fromPath = Segmenter::open('japanese', $path, null, null, self::USER_WORD_TAGS);
        $fromBytes = Segmenter::fromBytes('japanese', file_get_contents($path), null, null, self::USER_WORD_TAGS);
        $fromUri = Segmenter::fromUri('japanese', $path, null, null, self::USER_WORD_TAGS);
        // Named arguments work too, without $user_words.
        $named = Segmenter::open('japanese', $path, user_word_tags: self::USER_WORD_TAGS);

        $tokens = $fromPath->segmentWithPos($sentence);
        $this->assertSame('チャットGPT', $tokens[0]->surface);
        $this->assertSame('PROPN', $tokens[0]->pos);
        $this->assertSame(self::tagged($fromPath, $sentence), self::tagged($fromBytes, $sentence));
        $this->assertSame(self::tagged($fromPath, $sentence), self::tagged($fromUri, $sentence));
        $this->assertSame(self::tagged($fromPath, $sentence), self::tagged($named, $sentence));
        // Segmentation alone ignores the tags.
        $this->assertSame(['チャットGPT', 'に', '聞く'], $fromPath->segment($sentence));
    }

    public function testInvalidUserWordTagsThrow(): void
    {
        $path = self::modelPath('japanese_pos.model');
        try {
            Segmenter::open('japanese', $path, user_word_tags: ['ポケモン' => 'Noun']);
            $this->fail('expected InvalidArgumentException');
        } catch (InvalidArgumentException $error) {
            $this->assertStringContainsString('unknown UPOS tag "Noun"', $error->getMessage());
        }
        // A bad gate is still reported when only tagged words are given.
        try {
            Segmenter::open('japanese', $path, null, 1.5, self::USER_WORD_TAGS);
            $this->fail('expected InvalidArgumentException');
        } catch (InvalidArgumentException $error) {
            $this->assertMatchesRegularExpression('/0\.0\.\.=1\.0/', $error->getMessage());
        }
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
