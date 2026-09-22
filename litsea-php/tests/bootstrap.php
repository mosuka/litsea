<?php

declare(strict_types=1);

/**
 * PHPUnit bootstrap.
 *
 * The extension is loaded with `php -d extension=...`, so all this has to do
 * is fail loudly when it was not, rather than letting every test report a
 * confusing "class not found".
 */
// The module registers itself as `litsea` (see get_module in src/lib.rs): the
// same name PIE, `php -m` and `extension=litsea` use.
if (!extension_loaded('litsea')) {
    fwrite(
        STDERR,
        "The litsea extension is not loaded.\n"
        . "Run the tests through `make test-litsea-php`, or pass the built library:\n"
        . "  php -d extension=/path/to/liblitsea_php.so vendor/bin/phpunit -c litsea-php/phpunit.xml.dist\n"
    );
    exit(1);
}

// composer.json lives at the repository root (Packagist reads it there), so
// Composer installs into <root>/vendor rather than next to this directory.
require __DIR__ . '/../../vendor/autoload.php';
