param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$MinFreeGpuMiB = 9000,
    [int]$PollSeconds = 300,
    [switch]$SkipGpuWait,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$outputRoot = "outputs\junyi\sage_lite_v1\S3_bucketmlp_tail0p01_scope_true_e60_seed2025"
$checkpointRoot = "checkpoints\junyi\sage_lite_v1\S3_bucketmlp_tail0p01_scope_true_e60_seed2025"
$outputRootAbs = Join-Path $repoPath $outputRoot
$checkpointRootAbs = Join-Path $repoPath $checkpointRoot
$queueLog = Join-Path $outputRootAbs "junyi_sage_bucketmlp_tail0p01_queue.log"

New-Item -ItemType Directory -Force -Path $outputRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootAbs | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

function Get-GpuFreeMiB {
    try {
        $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            return [int](([string]$raw).Trim().Split("`n")[0].Trim())
        }
    } catch {
        return $null
    }
    return $null
}

Write-QueueLog "CONFIG Junyi SAGE bucket-MLP tail0p01 seed=2025 | output=$outputRoot | checkpoint=$checkpointRoot | gate_mode=bucket_mlp | buckets=20 | bucket_strategy=paper | gate_hidden=32 | ratio=0.01 | scope=true | mask=true | epochs=60"

if ($DryRun) {
    Write-QueueLog "DRYRUN requested; no training started."
    exit 0
}

if ($SkipGpuWait) {
    Write-QueueLog "SkipGpuWait requested; starting immediately."
} else {
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -eq $free) {
            Write-QueueLog "GPU free memory unavailable; wait ${PollSeconds}s."
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($free -ge $MinFreeGpuMiB) {
            Write-QueueLog "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB; starting."
            break
        }
        Write-QueueLog "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-QueueLog "START Junyi SAGE bucket-MLP tail0p01 seed=2025"

& .\run_junyi_sage_tailratio_grid_seed2025.ps1 `
    -Repo $repoPath `
    -Seed 2025 `
    -Ratios 0.01 `
    -Epochs 60 `
    -Patience 60 `
    -SagePoolTopK 64 `
    -SageGateMode bucket_mlp `
    -SageGateBuckets 20 `
    -SageGateHidden 32 `
    -SageGateBucketStrategy paper `
    -OutputRoot $outputRoot `
    -CheckpointRoot $checkpointRoot

$exitCode = $LASTEXITCODE
Write-QueueLog "END Junyi SAGE bucket-MLP tail0p01 seed=2025 | exit_code=$exitCode"
exit $exitCode
