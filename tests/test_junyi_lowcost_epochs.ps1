$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_junyi_main_table_lowcost_2026_2027.ps1"
$text = Get-Content -Raw -Encoding UTF8 -LiteralPath $script

if ($text -notmatch '\$env:LIGHTGCN_STATIC_EPOCHS = "60"') {
    throw "Junyi LightGCN should use 60 epochs; seed2025 peaked early and 300 is wasteful."
}

Write-Host "test_junyi_lowcost_epochs.ps1 passed"
