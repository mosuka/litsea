<?php

declare(strict_types=1);

/**
 * PHPUnit bootstrap.
 *
 * The extension is loaded with `php -d extension=...`, so all this has to do
 * is fail loudly when it was not, rather than letting every test report a
 * confusing "class not found".
 */
// ext-php-rs registers the extension under the crate name, hyphen and all.
if (!extension_loaded('litsea-php')) {
    fwrite(
        STDERR,
        "The litsea extension is not loaded.\n"
        . "Run the tests through `make test-litsea-php`, or pass the built library:\n"
        . "  php -d extension=/path/to/liblitsea_php.so vendor/bin/phpunit\n"
    );
    exit(1);
}

require __DIR__ . '/../vendor/autoload.php';
