<?php

declare(strict_types=1);

namespace Litsea\Tests;

use PHPUnit\Framework\TestCase;

/**
 * Shared helpers: model paths and the CLI the parity tests compare against.
 */
abstract class LitseaTestCase extends TestCase
{
    /** Repository root, three levels up from this file. */
    protected static function repoRoot(): string
    {
        return dirname(__DIR__, 2);
    }

    /** Absolute path to a bundled model. */
    protected static function modelPath(string $name): string
    {
        return self::repoRoot() . '/models/' . $name;
    }

    /**
     * Builds the `litsea` CLI once and returns the path to the binary.
     *
     * The parity tests compare against the CLI rather than hardcoded output,
     * so the reference implementation decides what is correct.
     */
    protected static function litseaCli(): string
    {
        $binary = self::repoRoot() . '/target/debug/litsea';
        if (!is_file($binary)) {
            exec(
                sprintf('cd %s && cargo build --quiet -p litsea-cli 2>&1', escapeshellarg(self::repoRoot())),
                $output,
                $status
            );
            if ($status !== 0) {
                self::fail("failed to build the litsea CLI: " . implode("\n", $output));
            }
        }

        return $binary;
    }

    /**
     * Runs the CLI over $input and returns its output lines.
     *
     * @param string[] $args
     * @return string[]
     */
    protected static function runCli(array $args, string $input): array
    {
        $command = escapeshellarg(self::litseaCli());
        foreach ($args as $arg) {
            $command .= ' ' . escapeshellarg($arg);
        }

        $descriptors = [0 => ['pipe', 'r'], 1 => ['pipe', 'w'], 2 => ['pipe', 'w']];
        $process = proc_open($command, $descriptors, $pipes);
        self::assertIsResource($process, 'failed to start the CLI');

        fwrite($pipes[0], $input);
        fclose($pipes[0]);
        $stdout = stream_get_contents($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[1]);
        fclose($pipes[2]);
        $status = proc_close($process);

        self::assertSame(0, $status, "the CLI failed: {$stderr}");

        return array_values(array_filter(explode("\n", $stdout), static fn ($line) => $line !== ''));
    }

    /** Creates a temporary directory for a test's artifacts. */
    protected static function tempDir(): string
    {
        $dir = sys_get_temp_dir() . '/litsea-php-' . bin2hex(random_bytes(6));
        mkdir($dir, 0o700, true);

        return $dir;
    }
}
