param(
    [string]$SourceManifest = "outputs\content_delta_pop5\sage_lite_v1\S8_tail0p002_e12_maskfalse\strict_item_cold_balanced_thr1_seed_2025\static_protocol_manifest.json",
    [string]$SourceCheckpointDir = "checkpoints\content_delta_pop5\sage_lite_v1\S8_tail0p002_e12_maskfalse\strict_item_cold_balanced_thr1_seed_2025",
    [string]$OutputRoot = "outputs\content_delta_pop5\sage_lite_v1\S9_tail0p002_maskfalse_e60_resume_from_s8e12",
    [string]$CheckpointRoot = "checkpoints\content_delta_pop5\sage_lite_v1\S9_tail0p002_maskfalse_e60_resume_from_s8e12",
    [int]$Seed = 2025,
    [int]$Threshold = 1,
    [int]$MinGpuFreeMiB = 4500,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Set-Location -Path $PSScriptRoot

$tag = "strict_item_cold_balanced_thr${Threshold}_seed_${Seed}"
$out = Join-Path $OutputRoot $tag
$ckpt = Join-Path $CheckpointRoot $tag
$runLog = Join-Path $out "run.log"
$launcherLog = Join-Path $out "launcher.log"

New-Item -ItemType Directory -Force -Path $out | Out-Null
New-Item -ItemType Directory -Force -Path $ckpt | Out-Null

function Write-LaunchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $launcherLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        return [int]($raw | Select-Object -First 1)
    } catch {
        return 0
    }
}

if (-not (Test-Path $SourceManifest)) {
    throw "Missing source manifest: $SourceManifest"
}

$sourceLatest = Join-Path $SourceCheckpointDir "latest.pt"
$sourceEpoch12 = Join-Path $SourceCheckpointDir "epoch_012.pt"
if (-not (Test-Path $sourceLatest)) {
    throw "Missing source latest checkpoint: $sourceLatest"
}

Copy-Item -LiteralPath $sourceLatest -Destination (Join-Path $ckpt "latest.pt") -Force
if (Test-Path $sourceEpoch12) {
    Copy-Item -LiteralPath $sourceEpoch12 -Destination (Join-Path $ckpt "epoch_012.pt") -Force
}

$manifest = Get-Content -LiteralPath $SourceManifest -Raw | ConvertFrom-Json
foreach ($prop in $manifest.env.PSObject.Properties) {
    if ($prop.Name -like "USIM_*") {
        Set-Item -Path "Env:$($prop.Name)" -Value ([string]$prop.Value)
    }
}

$env:USIM_FB_OUTPUT_DIR = $out
$env:USIM_FB_OUTPUT_TAG = $tag
$env:USIM_FB_CKPT_DIR = $ckpt
$env:USIM_FB_FORCE_FRESH = "0"
$env:USIM_FB_AUTO_RESUME = "1"
$env:USIM_FB_SAVE_CKPT = "1"
$env:USIM_FB_SAVE_OPT_STATE = "1"
$env:USIM_STATIC_SEED = [string]$Seed
$env:USIM_SEED = [string]$Seed
$env:USIM_COLD_THRESHOLD = [string]$Threshold
$env:USIM_N_EPOCHS = "60"
$env:USIM_EARLY_STOP_PATIENCE = "60"
$env:USIM_FB_SNAPSHOT_EPOCHS = "20,30,40,50,60"
$env:USIM_MASK_KNOWN_POS_NEG = "0"
$env:USIM_MASK_SAME_ITEM_NEG = "0"

Write-LaunchLog "Prepared S9 mask=false e60 resume from $SourceCheckpointDir"
Write-LaunchLog "Output=$out"
Write-LaunchLog "Checkpoint=$ckpt"
Write-LaunchLog "AutoResume=$env:USIM_FB_AUTO_RESUME ForceFresh=$env:USIM_FB_FORCE_FRESH Epochs=$env:USIM_N_EPOCHS Patience=$env:USIM_EARLY_STOP_PATIENCE"
Write-LaunchLog "MaskKnownPosNeg=$env:USIM_MASK_KNOWN_POS_NEG MaskSameItemNeg=$env:USIM_MASK_SAME_ITEM_NEG"

while ($true) {
    $free = Get-GpuFreeMiB
    if ($free -ge $MinGpuFreeMiB) {
        Write-LaunchLog "GPU free ${free}MiB >= ${MinGpuFreeMiB}MiB; starting run."
        break
    }
    Write-LaunchLog "Waiting for GPU free memory: ${free}MiB < ${MinGpuFreeMiB}MiB"
    Start-Sleep -Seconds $PollSeconds
}

Write-LaunchLog "Launching python training."
& ".\py.bat" -u -X faulthandler "usim_feedback_fast3_content_delta.py" 2>&1 | Tee-Object -FilePath $runLog
$exitCode = $LASTEXITCODE
Write-LaunchLog "Training finished with exit code $exitCode"
exit $exitCode
