$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_gar_coldrec_3seed_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing ColdRec GAR three-seed runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("gar_coldrec_3seed_" + [System.Guid]::NewGuid().ToString("N"))
$formalRoot = Join-Path $tmpRoot "formal"
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script -OutputRoot $formalRoot -DryRun *>&1
    $text = $out -join "`n"

    foreach ($needle in @(
        "ColdRec GAR strict source-default three-seed",
        "seeds=2025,2026,2027",
        "single_runner=run_gar_coldrec_single_seed.ps1",
        "MF epochs=500",
        "GAR epochs=500",
        "early_stop=5",
        "use_gpu=True",
        "strict_item_cold_balanced_thr1_seed_2025",
        "strict_item_cold_balanced_thr1_seed_2026",
        "strict_item_cold_balanced_thr1_seed_2027",
        "gar_coldrec_3seed_detail.csv",
        "gar_coldrec_3seed_summary.csv",
        "gar_coldrec_3seed_summary.json",
        "gar_coldrec_3seed_report.md",
        "aggregate_log=aggregate.log"
    )) {
        if ($text -notmatch [regex]::Escape($needle)) {
            throw "Expected dry-run output to contain '$needle'. Output:`n$text"
        }
    }

    $positions = foreach ($seed in @(2025, 2026, 2027)) {
        $needle = "TASK seed=$seed"
        $matches = [regex]::Matches($text, [regex]::Escape($needle))
        if ($matches.Count -ne 1) {
            throw "Expected exactly one '$needle', found $($matches.Count). Output:`n$text"
        }
        $matches[0].Index
    }
    if (-not ($positions[0] -lt $positions[1] -and $positions[1] -lt $positions[2])) {
        throw "Seeds are not printed in serial order. Output:`n$text"
    }
    if (Test-Path -LiteralPath $formalRoot) {
        throw "Dry-run unexpectedly created output root: $formalRoot"
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "GAR ColdRec three-seed runner contract: PASS"
