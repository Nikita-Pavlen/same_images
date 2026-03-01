<?php
require_once 'vendor/autoload.php';

use Gregwar\Image\Image;

$sourceDir = __DIR__ . '/py-images';
$resultDir = __DIR__ . '/php-result-images';

if (!is_dir($resultDir)) {
    mkdir($resultDir, 0777, true);
}

function compareImages(string $imgPath1, string $imgPath2): float
{
    $img1 = imagecreatefromjpeg($imgPath1);
    $img2 = imagecreatefromjpeg($imgPath2);

    if (!$img1 || !$img2) {
        return 0.0;
    }

    $totalDiff = 0;
    $pixelCount = 32 * 32;

    for ($x = 0; $x < 32; $x++) {
        for ($y = 0; $y < 32; $y++) {
            $rgb1 = imagecolorat($img1, $x, $y) & 0xFF;
            $rgb2 = imagecolorat($img2, $x, $y) & 0xFF;
            $totalDiff += abs($rgb1 - $rgb2);
        }
    }

    imagedestroy($img1);
    imagedestroy($img2);

    $maxDiff = $pixelCount * 255;
    return (1 - $totalDiff / $maxDiff) * 100;
}

function getMainImage(string $categoryDir): ?string
{
    $mainDir = $categoryDir . '/main';
    if (!is_dir($mainDir)) {
        return null;
    }
    $files = glob($mainDir . '/*.jpg');
    return $files ? $files[0] : null;
}

$images = glob($sourceDir . '/*.jpg');
$total = count($images);

echo "Found {$total} images to process\n";

$tempFile = tempnam(sys_get_temp_dir(), 'img_') . '.jpg';
$processed = 0;
$newCategories = 0;
$matched = 0;

foreach ($images as $imagePath) {
    $processed++;
    $filename = basename($imagePath);

    Image::open($imagePath)
        ->resize(32, 32)
        ->grayscale()
        ->save($tempFile);

    $bestMatch = null;
    $bestScore = 0;

    $categories = glob($resultDir . '/*', GLOB_ONLYDIR);
    foreach ($categories as $catDir) {
        $mainImage = getMainImage($catDir);
        if (!$mainImage) {
            continue;
        }

        $score = compareImages($tempFile, $mainImage);
        if ($score > $bestScore) {
            $bestScore = $score;
            $bestMatch = $catDir;
        }
    }

    if ($bestMatch !== null && $bestScore >= 90) {
        copy($imagePath, $bestMatch . '/' . $filename);
        $matched++;
        echo "[{$processed}/{$total}] {$filename} -> " . basename($bestMatch) . " (score: " . round($bestScore, 2) . "%)\n";
    } else {
        $catName = 'category_' . str_pad($newCategories + 1, 4, '0', STR_PAD_LEFT);
        $catDir = $resultDir . '/' . $catName;
        $mainDir = $catDir . '/main';

        mkdir($mainDir, 0777, true);
        copy($tempFile, $mainDir . '/' . $filename);
        copy($imagePath, $catDir . '/' . $filename);

        $newCategories++;
        echo "[{$processed}/{$total}] {$filename} -> NEW {$catName}\n";
    }
}

if (file_exists($tempFile)) {
    unlink($tempFile);
}

echo "\nDone! Processed: {$total}, Matched: {$matched}, New categories: {$newCategories}\n";
