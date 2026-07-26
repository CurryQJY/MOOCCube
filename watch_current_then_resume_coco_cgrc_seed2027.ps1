param(
    [int]$Seed = 2027,
    [int]$CheckSeconds = 60,
    [int]$Epochs = 50,
    [int]$BatchSize = 512,
    [int]$ReconUserChunk = 256,
    [int]$ReconTopK = 20
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$QueueDir = Join-Path $RepoRoot "outputs\coco\single_seed_triage\_queue"
$LogPath = Join-Path $QueueDir "watch_current_then_resume_coco_cgrc_seed${Seed}.log"
$SplitName = "strict_item_cold_balanced_thr1_seed_$Seed"
$ResultPath = Join-Path $RepoRoot "outputs\coco\single_seed_triage\ours_full\$SplitName\main_table_compare\cgrc_paper_static_result.json"
$CkptPath = Join-Path $RepoRoot "checkpoints\coco\single_seed_triage\cgrc_paper\$SplitName\latest.pt"
$Runner = Join-Path $RepoRoot "run_coco_cgrc_paper_single_seed.ps1"

New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

function Write-WatchLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $LogPath -Encoding UTF8 -Value $line
    Write-Host $line
}

function Get-BlockingTrainingProcesses {
    $allowedNames = @(
        "python.exe",
        "pythonw.exe",
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe"
    )
    $patterns = @(
        "usim_feedback_fast3_content_delta.py",
        "usim_feedback_fast3_content_delta_v3.py",
        "cgrc_paper_static_hin.py",
        "aldi_static_hin.py",
        "bpr_static",
        "lightgcn_static",
        "drop_static_hin.py",
        "gar_static_hin.py",
        "ccfc_static_hin.py",
        "lightgcl_static",
        "sagerec_static",
        "course_aware_mlp",
        "run_xes3g5m_ours_sota_serial.ps1",
        "run_xes3g5m_lightweight_baselines.ps1",
        "run_coco_cgrc_paper_single_seed.ps1",
        "run_coco_missing_two_seed_main_table_serial.ps1"
    )
    Get-CimInstance Win32_Process | Where-Object {
        if ($_.ProcessId -eq $PID) {
            return $false
        }
        $name = [string]$_.Name
        if ($allowedNames -notcontains $name.ToLowerInvariant()) {
            return $false
        }
        $cmd = [string]$_.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) {
            return $false
        }
        foreach ($pattern in $patterns) {
            if ($cmd -like "*$pattern*") {
                return $true
            }
        }
        return $false
    }
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Missing runner: $Runner"
}
if (-not (Test-Path -LiteralPath $CkptPath)) {
    throw "Missing checkpoint for resume: $CkptPath"
}
if (Test-Path -LiteralPath $ResultPath) {
    Write-WatchLog "result already exists; no resume needed: $ResultPath"
    exit 0
}

Write-WatchLog "watch start seed=$Seed target_epochs=$Epochs batch=$BatchSize chunk=$ReconUserChunk topk=$ReconTopK"
Write-WatchLog "checkpoint=$CkptPath"

while ($true) {
    if (Test-Path -LiteralPath $ResultPath) {
        Write-WatchLog "result appeared while waiting; no resume needed: $ResultPath"
        exit 0
    }

    $running = @(Get-BlockingTrainingProcesses)
    if ($running.Count -eq 0) {
        break
    }

    $summary = ($running | Select-Object ProcessId, Name, CommandLine |
        ForEach-Object {
            $cmd = [string]$_.CommandLine
            if ($cmd.Length -gt 160) {
                $cmd = $cmd.Substring(0, 160) + "..."
            }
            "{0}:{1}:{2}" -f $_.ProcessId, $_.Name, $cmd
        }) -join " | "
    Write-WatchLog "waiting current training processes=$($running.Count) :: $summary"
    Start-Sleep -Seconds $CheckSeconds
}

Write-WatchLog "no blocking training process; start CGRC resume seed=$Seed"
& $Runner -Seed $Seed -Epochs $Epochs -BatchSize $BatchSize -ReconUserChunk $ReconUserChunk -ReconTopK $ReconTopK
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-WatchLog "CGRC resume failed exit=$exitCode"
    exit $exitCode
}

if (Test-Path -LiteralPath $ResultPath) {
    Write-WatchLog "CGRC resume complete result=$ResultPath"
} else {
    Write-WatchLog "CGRC runner exited 0 but result is missing: $ResultPath"
    exit 2
}
