#!/usr/bin/env php
<?php

declare(strict_types=1);

/**
 * Prints stubs for the loaded litsea extension to stdout.
 *
 * Usage (from the repository root, after `cargo build -p litsea-php`):
 *
 *   php -n -d extension=target/debug/liblitsea_php.so litsea-php/tools/generate-stubs.php \
 *       > litsea-php/stubs/litsea.stubs.php
 *
 * `make stubs-litsea-php` does exactly that. The committed stub file must
 * match this output; `StubsTest` checks that.
 */

require __DIR__ . '/StubGenerator.php';

if (!extension_loaded('litsea')) {
    fwrite(STDERR, "The litsea extension is not loaded; pass it with -d extension=/path/to/liblitsea_php.so\n");
    exit(1);
}

echo (new Litsea\Tools\StubGenerator())->generate();
