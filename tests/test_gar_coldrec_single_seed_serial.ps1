$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_gar_coldrec_single_seed.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing ColdRec GAR single-seed runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("gar_coldrec_single_seed_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script `
        -OutputDir (Join-Path $tmpRoot "result") `
        -DryRun *>&1
    $text = ($out -join "`n")

    foreach ($needle in @(
        "dataset=mooccube seed=2025",
        "strict_item_cold_balanced_thr1_seed_2025",
        "processed_data_hin_clean_pop5",
        "gar_coldrec_static.py",
        "MF epochs=5",
        "GAR epochs=10",
        "train_only",
        "use_gpu=True",
        "STAGE 1 MF",
        "STAGE 2 GAR",
        "gar_coldrec_strict_result.json"
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

Write-Host "GAR ColdRec single-seed runner contract: PASS"
