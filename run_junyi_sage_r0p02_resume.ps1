param(
    [string]$Repo = "D:\DeskTop\MOOCCube"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$queueLog = Join-Path $Repo "outputs\junyi\sage_lite_v1\S1_tailratio_grid_seed2025\_queue\tailratio_grid_queue.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $queueLog) | Out-Null
Add-Content -LiteralPath $queueLog -Encoding UTF8 -Value ("[{0}] RESUME START ratio=0.02 | AutoResume=True | ForceFresh=False" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

try {
    & (Join-Path $Repo "run_usim_feedback_fast3_content_delta_static.ps1") `
        -PythonRunner ".\py.bat" `
        -DataDir "processed_data_junyi" `
        -RelationDir "processed_data_junyi\relations" `
        -OutputRoot "outputs\junyi\sage_lite_v1\S1_tailratio_grid_seed2025\r0p02" `
        -CheckpointRoot "checkpoints\junyi\sage_lite_v1\S1_tailratio_grid_seed2025\r0p02" `
        -Protocol strict_item_cold_balanced `
        -ColdThresholds 1 `
        -Seeds 2025 `
        -Epochs 60 `
        -Patience 60 `
        -EarlyStopAverageMode item_macro `
        -UseContentDelta:$false `
        -UsePseudoColdTrain:$false `
        -UsePaac:$false `
        -CourseFeedbackOnlyCold:$true `
        -CourseSampleOnlyCold:$true `
        -PrereqAuxOnlyCold:$true `
        -RunSampledEval:$false `
        -UseSageLite:$true `
        -SageOnlyColdOrTail:$true `
        -SageTailPopRatio 0.02 `
        -SagePoolTopK 64 `
        -SageUseTwoExpert:$false `
        -UseSageAuxLoss:$false `
        -MaskKnownPosNeg:$true `
        -MaskSameItemNeg:$true `
        -SaveCkpt:$true `
        -AutoResume:$true `
        -ForceFresh:$false `
        -SaveOptState:$true `
        -SkipAggregate

    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
    Add-Content -LiteralPath $queueLog -Encoding UTF8 -Value ("[{0}] RESUME END ratio=0.02 | exit={1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $exitCode)
    exit $exitCode
} catch {
    Add-Content -LiteralPath $queueLog -Encoding UTF8 -Value ("[{0}] RESUME ERROR ratio=0.02 | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $_.Exception.Message)
    throw
}
