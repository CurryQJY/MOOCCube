param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [ValidateRange(10, 300)]
    [int]$PollIntervalSec = 60,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

$upstreamManifestPath = "outputs\cbi_anchor_sim_3seed_serial\run_manifest.json"
$screenScript = "run_cbi_anchor_aux_screen_seed2025.ps1"
$queueRoot = "outputs\cbi_anchor_aux_screen_seed2025"
$queueManifestPath = Join-Path $queueRoot "queue_manifest.json"
$queueLogRoot = "background_logs\cbi_anchor_aux_screen_seed2025"
$queueLogPath = Join-Path $queueLogRoot "queue.log"

function Write-QueueManifest($Payload) {
    New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
    $Payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $queueManifestPath -Encoding UTF8
}

if ($DryRun) {
    Write-Host "DRY_RUN wait for CBI anchor three-seed then auxiliary screen"
    Write-Host ("Upstream={0}" -f $upstreamManifestPath)
    Write-Host ("Screen={0}" -f $screenScript)
    Write-Host ("PollIntervalSec={0}" -f $PollIntervalSec)
    exit 0
}

New-Item -ItemType Directory -Force -Path $queueRoot, $queueLogRoot | Out-Null
$queue = [ordered]@{
    schema_version = 1
    experiment = "wait_cbi_anchor_3seed_then_aux_screen"
    status = "waiting"
    upstream_manifest = [IO.Path]::GetFullPath((Join-Path $repoPath $upstreamManifestPath))
    screen_script = [IO.Path]::GetFullPath((Join-Path $repoPath $screenScript))
    poll_interval_seconds = $PollIntervalSec
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    completed_at_utc = $null
    upstream_status = $null
    error = $null
}
Write-QueueManifest $queue

while ($true) {
    if (-not (Test-Path -LiteralPath $upstreamManifestPath)) {
        "[$(Get-Date -Format o)] waiting: upstream manifest missing" | Tee-Object -FilePath $queueLogPath -Append
        Start-Sleep -Seconds $PollIntervalSec
        continue
    }

    $upstream = Get-Content -LiteralPath $upstreamManifestPath -Raw | ConvertFrom-Json
    $queue.upstream_status = [string]$upstream.status
    Write-QueueManifest $queue

    if ($upstream.status -eq "completed") {
        $queue.status = "launching"
        Write-QueueManifest $queue
        "[$(Get-Date -Format o)] upstream completed; launching auxiliary screen" | Tee-Object -FilePath $queueLogPath -Append
        try {
            & (Join-Path $repoPath $screenScript) -Repo $repoPath *>&1 | Tee-Object -FilePath $queueLogPath -Append
            if (-not $?) {
                throw "Auxiliary screen returned an unsuccessful status."
            }
            $queue.status = "completed"
        }
        catch {
            $queue.status = "failed"
            $queue.error = $_.Exception.Message
        }
        $queue.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        Write-QueueManifest $queue
        if ($queue.status -eq "failed") {
            throw $queue.error
        }
        exit 0
    }

    if ($upstream.status -eq "failed") {
        $queue.status = "upstream_failed"
        $queue.error = [string]$upstream.error
        $queue.completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        Write-QueueManifest $queue
        throw "Upstream three-seed experiment failed: $($upstream.error)"
    }

    "[$(Get-Date -Format o)] waiting: upstream status=$($upstream.status)" | Tee-Object -FilePath $queueLogPath -Append
    Start-Sleep -Seconds $PollIntervalSec
}
