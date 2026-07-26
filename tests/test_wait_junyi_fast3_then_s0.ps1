$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "wait_junyi_fast3_then_run_s0.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing watcher script: $script"
}

$tmp = Join-Path $repo ".runtime_tmp\test_wait_junyi_fast3_then_s0"
Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$junyiRoot = Join-Path $tmp "junyi"
foreach ($seed in @(2026, 2027)) {
    $dir = Join-Path $junyiRoot ("strict_item_cold_balanced_thr1_seed_{0}" -f $seed)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Set-Content -LiteralPath (Join-Path $dir "final_fullrank_usim_feedback_fast3_content_delta_static.csv") `
        -Encoding UTF8 `
        -Value "model,protocol,full_cold_item_macro_n10`nmock,static,0.1"
}

$s0Root = Join-Path $tmp "s0_outputs"
$s0Ckpt = Join-Path $tmp "s0_checkpoints"

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -JunyiOutputRoot $junyiRoot `
    -JunyiSeeds 2026,2027 `
    -S0OutputRoot $s0Root `
    -S0CheckpointRoot $s0Ckpt `
    -S0Seed 2025 `
    -PollSeconds 1 `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "JUNYI FAST3 DONE seed=2026",
    "JUNYI FAST3 DONE seed=2027",
    "DRYRUN START S0 seed=2025",
    "UseSageLite=True",
    "SagePoolTopK=48"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected dry-run line: $expected"
    }
}

Write-Host "test_wait_junyi_fast3_then_s0.ps1 passed"
