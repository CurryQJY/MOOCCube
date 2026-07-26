$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repo

$outputDir = "outputs\cbi_faithful_seed2025_eval_all_refined"
$logPath = Join-Path $outputDir "evaluation.log"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

& .\py.bat .\evaluate_cbi_all_refined_seed2025.py *>&1 |
    Tee-Object -FilePath $logPath

if ($LASTEXITCODE -ne 0) {
    throw "CBI all-refined evaluation failed with exit code $LASTEXITCODE"
}
