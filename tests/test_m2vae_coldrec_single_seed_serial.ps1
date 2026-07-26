$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_m2vae_coldrec_single_seed_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing ColdRec M2VAE single-seed runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("m2vae_coldrec_single_seed_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script `
        -OutputRoot (Join-Path $tmpRoot "outputs") `
        -DryRun *>&1
    $text = ($out -join "`n")

    if ($text -notmatch "Total tasks: 3") {
        throw "Expected dry-run to list exactly 3 tasks. Output:`n$text"
    }
    foreach ($needle in @(
        "dataset=mooccube seed=2025",
        "dataset=junyi seed=2025",
        "dataset=coco seed=2025",
        "processed_data_hin_clean_pop5",
        "processed_data_junyi",
        "processed_data_coco",
        "m2vae_coldrec_static.py",
        ".runtime_tmp\ColdRec",
        "MF epochs=",
        "M2VAE epochs="
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

Write-Host "test_m2vae_coldrec_single_seed_serial.ps1 passed"
