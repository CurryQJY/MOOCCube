param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$MaxParallel = 3,
    [int]$PollSeconds = 30,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

if ($MaxParallel -lt 1 -or $MaxParallel -gt 3) {
    throw "MaxParallel must be between 1 and 3 for the 12 GB GPU."
}

$runner = (Resolve-Path '.\run_learner_guided_full_seed2025.ps1').Path
$queueRoot = "background_logs\lira_v2_ablation_complete_3seed"
New-Item -ItemType Directory -Path $queueRoot -Force | Out-Null
$queueLog = Join-Path $queueRoot "queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $queueLog -Append
}

function Format-Number([double]$Value) {
    return $Value.ToString([Globalization.CultureInfo]::InvariantCulture)
}

$arms = @(
    [ordered]@{ Key="ablation_t0"; Steps=0; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ Key="ablation_t1"; Steps=1; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ Key="ablation_no_stop"; Steps=3; MinFit=0.0; MinGain=0.0; RefineW=0.5; StableW=0.01 },
    [ordered]@{ Key="ablation_no_refined_loss"; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ Key="ablation_no_stability"; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.0 }
)
$seeds = @(2026,2027)

Write-QueueLog "WAIT seed2025 reference ablations"
$seed2025Checkpoints = @($arms | ForEach-Object {
    "checkpoints\learner_guided_full\lira_v2_$($_.Key)_seed2025\strict_item_cold_balanced_thr1_seed_2025\validation_finished.pt"
})
while ($true) {
    $missing = @($seed2025Checkpoints | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -eq 0) { break }
    Write-QueueLog ("WAIT missing={0}" -f ($missing -join ','))
    if ($DryRun) { throw "DryRun cannot continue because seed2025 reference ablations are incomplete." }
    Start-Sleep -Seconds $PollSeconds
}

$jobs = @()
foreach ($arm in $arms) {
    foreach ($seed in $seeds) {
        $runName = "lira_v2_$($arm.Key)_seed$seed"
        $checkpoint = "checkpoints\learner_guided_full\$runName\strict_item_cold_balanced_thr1_seed_$seed\validation_finished.pt"
        $jobs += [pscustomobject]@{
            Arm = $arm
            Seed = $seed
            RunName = $runName
            Checkpoint = $checkpoint
        }
    }
}

$pending = @()
foreach ($job in $jobs) {
    if (Test-Path -LiteralPath $job.Checkpoint) {
        Write-QueueLog "SKIP $($job.RunName) completed"
    }
    else {
        $pending += $job
    }
}

if ($DryRun) {
    foreach ($job in $pending) {
        Write-QueueLog "DRYRUN $($job.RunName)"
    }
    Write-QueueLog "DRYRUN DONE pending=$($pending.Count)"
    exit 0
}

for ($start = 0; $start -lt $pending.Count; $start += $MaxParallel) {
    $end = [Math]::Min($start + $MaxParallel - 1, $pending.Count - 1)
    $wave = @($pending[$start..$end])
    $launched = @()

    foreach ($job in $wave) {
        $stdout = Join-Path $queueRoot "$($job.RunName).stdout.log"
        $stderr = Join-Path $queueRoot "$($job.RunName).stderr.log"
        $arm = $job.Arm
        $args = @(
            '-NoProfile','-ExecutionPolicy','Bypass','-File',$runner,
            '-Seed',[string]$job.Seed,
            '-Epochs','35','-Patience','10',
            '-UsimSteps',[string]$arm.Steps,
            '-MinFit',(Format-Number $arm.MinFit),
            '-MinGain',(Format-Number $arm.MinGain),
            '-RefinementLossWeight',(Format-Number $arm.RefineW),
            '-StabilityLossWeight',(Format-Number $arm.StableW),
            '-RunName',$job.RunName
        )
        $process = Start-Process powershell.exe -ArgumentList $args -WorkingDirectory (Get-Location) `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $launched += [pscustomobject]@{ Process=$process; Job=$job; Stderr=$stderr }
        Write-QueueLog "START $($job.RunName) pid=$($process.Id)"
    }

    @($launched | ForEach-Object { $_.Process }) | Wait-Process
    foreach ($entry in $launched) {
        $entry.Process.Refresh()
        if ($entry.Process.ExitCode -ne 0) {
            throw "Ablation process failed: $($entry.Job.RunName), exit=$($entry.Process.ExitCode), stderr=$($entry.Stderr)"
        }
        $checkpoint = $entry.Job.Checkpoint
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            throw "Ablation failed or incomplete: $($entry.Job.RunName)"
        }
        Write-QueueLog "DONE $($entry.Job.RunName)"
    }
}

Write-QueueLog "ALL 3-SEED ABLATIONS DONE"
