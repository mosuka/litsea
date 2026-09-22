<?php

declare(strict_types=1);

namespace Litsea\Tests;

use Litsea\Tools\StubGenerator;
use PhpToken;
use PHPUnit\Framework\TestCase;
use ReflectionExtension;

/**
 * Guards the extension name and the committed stub file.
 *
 * The stub file is generated from the compiled extension, so the strongest
 * check is simply "regenerating it yields the committed bytes". The other
 * tests do not go through the generator, so a generator bug that dropped a
 * class would still be caught.
 */
final class StubsTest extends TestCase
{
    private const STUB = __DIR__ . '/../stubs/litsea.stubs.php';

    public function testExtensionRegistersAsLitsea(): void
    {
        // PIE's extension-name, php -m, extension=litsea and litsea.so all
        // use this name; the crate name litsea-php must not leak through.
        self::assertTrue(extension_loaded('litsea'));
        self::assertFalse(extension_loaded('litsea-php'));
        self::assertSame('litsea', (new ReflectionExtension('litsea'))->getName());
    }

    public function testStubFileMatchesTheCompiledExtension(): void
    {
        self::assertStringEqualsFile(
            self::STUB,
            (new StubGenerator())->generate(),
            'litsea-php/stubs/litsea.stubs.php is out of date; run `make stubs-litsea-php`.',
        );
    }

    public function testStubFileParses(): void
    {
        // TOKEN_PARSE makes the tokenizer throw ParseError on invalid PHP.
        $tokens = PhpToken::tokenize((string) file_get_contents(self::STUB), TOKEN_PARSE);

        self::assertNotEmpty($tokens);
        self::assertSame('<?php', trim($tokens[0]->text));
    }

    public function testStubDeclaresEveryClassAndFunction(): void
    {
        $extension = new ReflectionExtension('litsea');
        $code = (string) file_get_contents(self::STUB);

        self::assertStringContainsString("\nnamespace Litsea;\n", $code);

        $classes = $extension->getClassNames();
        self::assertNotEmpty($classes);
        foreach ($classes as $class) {
            $short = substr($class, strrpos($class, '\\') + 1);
            self::assertMatchesRegularExpression(
                '/^(?:(?:final|abstract) )?class ' . preg_quote($short, '/') . '\b/m',
                $code,
                "class {$class} is missing from the stub file",
            );
        }

        $functions = array_keys($extension->getFunctions());
        self::assertNotEmpty($functions);
        foreach ($functions as $function) {
            $short = substr($function, strrpos($function, '\\') + 1);
            self::assertMatchesRegularExpression(
                '/^function ' . preg_quote($short, '/') . '\(/m',
                $code,
                "function {$function} is missing from the stub file",
            );
        }
    }
}
