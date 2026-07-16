param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$Epochs = 15
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$logRoot = "background_logs\learner_guided_seed2025_core_screen"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$arms = @(
    [ordered]@{ Name = "lira_clean_t0_seed2025"; Steps = 0 },
    [ordered]@{ Name = "lira_one_step_t1_seed2025"; Steps = 1 },
    [ordered]@{ Name = "lira_full_t3_seed2025"; Steps = 3 }
)

foreach ($arm in $arms) {
    $log = Join-Path $logRoot ("{0}.log" -f $arm.Name)
    "===== START $($arm.Name) $(Get-Date -Format o) =====" | Tee-Object -FilePath $log
    & .\run_learner_guided_full_seed2025.ps1 `
        -Seed 2025 -Epochs $Epochs -Patience $Epochs `
        -UsimSteps $arm.Steps -RunName $arm.Name -ForceFresh `
        *>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Core screen arm $($arm.Name) failed with exit code $LASTEXITCODE"
    }
    "===== DONE $($arm.Name) $(Get-Date -Format o) =====" | Tee-Object -FilePath $log -Append
}
