param(
    [string]$RepoRoot = "D:\DeskTop\MOOCCube",
    [string]$WaitRunName = "S2_gatefix_tail0p01_e5_seed2025",
    [string]$WaitRunLog = "outputs\mooccubex\sage_lite_v1\S2_gatefix_tail0p01_e5_seed2025\strict_item_cold_balanced_thr1_seed_2025\run.log",
    [string]$OutputRoot = "outputs\mooccubex\relations_aug_v2_e30",
    [string]$CheckpointRoot = "checkpoints\mooccubex\relations_aug_v2_e30",
    [int[]]$Seeds = @(2026, 2027),
    [int]$Epochs = 30,
    [int]$Patience = 15,
    [int]$PollSeconds = 300,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

$queueDir = Join-Path $RepoRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$queueLog = Join-Path $queueDir "wait_s2_then_augv2_missing_seeds.log"

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $queueLog -Append
}

function Get-WaitRunProcesses {
    $pattern = [regex]::Escape($WaitRunName)
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -and
            $_.CommandLine -match $pattern -and
            $_.CommandLine -notmatch "wait_s2_then_run_mooccubex_augv2_missing_seeds" -and
            $_.CommandLine -notmatch "Get-CimInstance"
        } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

function Test-WaitRunCompletedCleanly {
    $absoluteWaitLog = Join-Path $RepoRoot $WaitRunLog
    $waitRunDir = Split-Path -Parent $absoluteWaitLog
    $finalFullRank = Join-Path $waitRunDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"

    if (Test-Path -LiteralPath $finalFullRank) {
        return $true
    }
    if (Test-Path -LiteralPath $absoluteWaitLog) {
        $savedLine = Select-String -Path $absoluteWaitLog -Pattern ">> Saved " -SimpleMatch -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($null -ne $savedLine) {
            return $true
        }
    }
    return $false
}

$seedText = ($Seeds -join ",")
Write-QueueLog "Watcher configured. WaitRun=$WaitRunName; seeds=$seedText; epochs=$Epochs; patience=$Patience; output=$OutputRoot; checkpoint=$CheckpointRoot."

if ($DryRun) {
    Write-QueueLog "DryRun only. Would wait for '$WaitRunName', then launch MOOCCubeX relations_aug_v2 missing seeds serially."
    return
}

while ($true) {
    $running = @(Get-WaitRunProcesses)
    if ($running.Count -eq 0) {
        $clean = Test-WaitRunCompletedCleanly
        if ($clean) {
            Write-QueueLog "Wait run appears completed cleanly. Launching augv2 missing seeds."
        } else {
            Write-QueueLog "No matching wait-run process remains, but no final marker was found. Launching augv2 anyway to avoid wasting the overnight window."
        }
        break
    }

    $pidText = (($running | ForEach-Object { "$($_.Name):$($_.ProcessId)" }) -join ", ")
    Write-QueueLog "Still waiting for $WaitRunName. Matching processes: $pidText."
    Start-Sleep -Seconds $PollSeconds
}

$oldConceptMin = $env:USIM_FB_COURSE_CONCEPT_MIN

try {
    $env:USIM_FB_COURSE_CONCEPT_MIN = "0.01"
    Write-QueueLog "Set USIM_FB_COURSE_CONCEPT_MIN=0.01 for augmented-relation run."
    Write-QueueLog "Starting serial augv2 run for seeds: $seedText."

    .\run_usim_feedback_fast3_content_delta_static.ps1 `
        -DataDir "processed_data_hin_x" `
        -RelationDir "MOOCCubeX\relations_aug_v2" `
        -Protocol strict_item_cold_balanced `
        -OutputRoot $OutputRoot `
        -CheckpointRoot $CheckpointRoot `
        -ColdThresholds 1 `
        -Seeds $Seeds `
        -Epochs $Epochs `
        -Patience $Patience `
        -EarlyStopAverageMode item_macro `
        -RunSampledEval $false `
        -UseContentDelta $false `
        -UsePseudoColdTrain $false `
        -UsePaac $false `
        -CourseFeedbackOnlyCold $false `
        -CourseSampleOnlyCold $false `
        -PrereqAuxOnlyCold $false `
        -SaveCkpt $true `
        -AutoResume $true `
        -ForceFresh $false `
        -SaveOptState $true

    Write-QueueLog "Augv2 missing-seed run finished."
} catch {
    Write-QueueLog "Augv2 missing-seed run failed: $($_.Exception.Message)"
    throw
} finally {
    if ($null -eq $oldConceptMin) {
        Remove-Item Env:USIM_FB_COURSE_CONCEPT_MIN -ErrorAction SilentlyContinue
    } else {
        $env:USIM_FB_COURSE_CONCEPT_MIN = $oldConceptMin
    }
}
