$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_usim_official_3datasets_3seed_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing official USIM serial runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("usim_official_serial_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script `
        -OutputRoot (Join-Path $tmpRoot "outputs") `
        -CheckpointRoot (Join-Path $tmpRoot "checkpoints") `
        -DryRun *>&1
    $text = ($out -join "`n")

    if ($text -notmatch "Total tasks: 9") {
        throw "Expected dry-run to list exactly 9 tasks. Output:`n$text"
    }
    foreach ($needle in @(
        "dataset=mooccube seed=2025",
        "dataset=mooccube seed=2026",
        "dataset=mooccube seed=2027",
        "dataset=junyi seed=2025",
        "dataset=junyi seed=2026",
        "dataset=junyi seed=2027",
        "dataset=coco seed=2025",
        "dataset=coco seed=2026",
        "dataset=coco seed=2027",
        "processed_data_hin_clean_pop5",
        "processed_data_junyi",
        "processed_data_coco"
    )) {
        if ($text -notmatch [regex]::Escape($needle)) {
            throw "Expected dry-run output to contain '$needle'. Output:`n$text"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_usim_official_3dataset_serial.ps1 passed"
