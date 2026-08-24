param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

$formalRuns = 2025,2026,2027 | ForEach-Object {
    "checkpoints\learner_guided_full\lira_v2_dynamic_dualloss_seed$_\strict_item_cold_balanced_thr1_seed_$_\validation_finished.pt"
}
$queueRoot = "background_logs\lira_v2_postformal_ablation_queue"
New-Item -ItemType Directory -Path $queueRoot -Force | Out-Null
$queueLog = Join-Path $queueRoot "queue.log"

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $queueLog -Append
}

Write-QueueLog "WAIT formal 3-seed validation checkpoints"
while ($true) {
    $missing = @($formalRuns | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -eq 0) { break }
    Write-QueueLog ("WAIT missing={0}" -f ($missing -join ","))
    Start-Sleep -Seconds $PollSeconds
}
Write-QueueLog "FORMAL DONE; start seed2025 core ablations"

$ablations = @(
    [ordered]@{ Name="lira_v2_ablation_t0_seed2025"; Steps=0; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ Name="lira_v2_ablation_t1_seed2025"; Steps=1; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.01 },
    [ordered]@{ Name="lira_v2_ablation_no_stop_seed2025"; Steps=3; MinFit=0.0; MinGain=0.0; RefineW=0.5; StableW=0.01 },
    [ordered]@{ Name="lira_v2_ablation_no_refined_loss_seed2025"; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.0; StableW=0.01 },
    [ordered]@{ Name="lira_v2_ablation_no_stability_seed2025"; Steps=3; MinFit=0.15; MinGain=0.001; RefineW=0.5; StableW=0.0 }
)

for ($start = 0; $start -lt $ablations.Count; $start += 3) {
    $wave = @($ablations[$start..([Math]::Min($start + 2, $ablations.Count - 1))])
    $processes = @()
    foreach ($arm in $wave) {
        $stdout = Join-Path $queueRoot "$($arm.Name).stdout.log"
        $stderr = Join-Path $queueRoot "$($arm.Name).stderr.log"
        $args = @(
            '-NoProfile','-ExecutionPolicy','Bypass','-File',
            (Resolve-Path '.\run_learner_guided_full_seed2025.ps1'),
            '-Seed','2025','-Epochs','35','-Patience','10',
            '-UsimSteps',[string]$arm.Steps,
            '-MinFit',[string]$arm.MinFit,
            '-MinGain',[string]$arm.MinGain,
            '-RefinementLossWeight',[string]$arm.RefineW,
            '-StabilityLossWeight',[string]$arm.StableW,
            '-RunName',$arm.Name,'-ForceFresh'
        )
        $process = Start-Process powershell.exe -ArgumentList $args -WorkingDirectory (Get-Location) `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $processes += $process
        Write-QueueLog "START $($arm.Name) pid=$($process.Id)"
    }
    $processes | Wait-Process
    foreach ($arm in $wave) {
        $checkpoint = "checkpoints\learner_guided_full\$($arm.Name)\strict_item_cold_balanced_thr1_seed_2025\validation_finished.pt"
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            throw "Ablation failed or incomplete: $($arm.Name)"
        }
        Write-QueueLog "DONE $($arm.Name)"
    }
}

Write-QueueLog "ALL TRAINING ABLATIONS DONE"
