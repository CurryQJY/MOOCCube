param(
    [int]$Seed = 2025,
    [int]$CheckSeconds = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$SplitName = "strict_item_cold_balanced_thr1_seed_$Seed"
$QueueDir = Join-Path $RepoRoot "outputs\coco\single_seed_triage\_queue"
$CompareDir = Join-Path $RepoRoot "outputs\coco\single_seed_triage\ours_full\$SplitName\main_table_compare"
$RunLog = Join-Path $CompareDir "run_cgrc_paper.log"
$ResultPath = Join-Path $CompareDir "cgrc_paper_static_result.json"
$MonitorLog = Join-Path $QueueDir "monitor_coco_cgrc_auto_downgrade.log"
$StatePath = Join-Path $QueueDir "monitor_coco_cgrc_auto_downgrade_state.json"

New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

function Get-Tiers {
    @(
        [pscustomobject]@{ Index = 0; Batch = 1024; Chunk = 1024; Name = "batch1024_chunk1024" }
        [pscustomobject]@{ Index = 1; Batch = 768;  Chunk = 768;  Name = "batch768_chunk768" }
        [pscustomobject]@{ Index = 2; Batch = 512;  Chunk = 512;  Name = "batch512_chunk512" }
        [pscustomobject]@{ Index = 3; Batch = 512;  Chunk = 256;  Name = "batch512_chunk256" }
    )
}

function Write-Monitor([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $MonitorLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-OomText([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }
    return ($Text -match "(?i)(CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|out of memory)")
}

function Read-SharedFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8, $true)
            try {
                return $reader.ReadToEnd()
            } finally {
                $reader.Dispose()
            }
        } finally {
            $stream.Dispose()
        }
    } catch {
        return ""
    }
}

function Get-CgrcProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) {
            return $false
        }
        if ($_.ProcessId -eq $PID) {
            return $false
        }
        (
            ($_.Name -eq "python.exe" -and $cmd -like "*cgrc_paper_static_hin.py*") -or
            ($_.Name -eq "cmd.exe" -and $cmd -like "*cgrc_paper_static_hin.py*") -or
            ($_.Name -eq "powershell.exe" -and $cmd -like "*-File*run_coco_cgrc_paper_single_seed.ps1*")
        )
    }
}

function Save-State($Tier) {
    [pscustomobject]@{
        tier_index = $Tier.Index
        batch = $Tier.Batch
        recon_user_chunk = $Tier.Chunk
        tier_name = $Tier.Name
        updated_at = (Get-Date).ToString("s")
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Start-Tier($Tier) {
    Save-State $Tier
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $QueueDir ("auto_coco_cgrc_{0}_{1}.out.log" -f $Tier.Name, $stamp)
    $err = Join-Path $QueueDir ("auto_coco_cgrc_{0}_{1}.err.log" -f $Tier.Name, $stamp)
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $RepoRoot "run_coco_cgrc_paper_single_seed.ps1"),
        "-Seed", "$Seed",
        "-Epochs", "50",
        "-BatchSize", "$($Tier.Batch)",
        "-ReconUserChunk", "$($Tier.Chunk)",
        "-ReconTopK", "20"
    )
    $proc = Start-Process -FilePath powershell.exe -ArgumentList $args -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Write-Monitor ("started tier={0} batch={1} chunk={2} pid={3}" -f $Tier.Index, $Tier.Batch, $Tier.Chunk, $proc.Id)
}

$tiers = Get-Tiers
$tierIndex = 0
if (Test-Path -LiteralPath $StatePath) {
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        $tierIndex = [int]$state.tier_index
    } catch {
        $tierIndex = 0
    }
}
if ($tierIndex -lt 0 -or $tierIndex -ge $tiers.Count) {
    $tierIndex = 0
}
Save-State $tiers[$tierIndex]
Write-Monitor ("monitor start seed={0} tier={1} check_seconds={2}" -f $Seed, $tierIndex, $CheckSeconds)

while ($true) {
    if (Test-Path -LiteralPath $ResultPath) {
        Write-Monitor ("result exists; monitor complete: {0}" -f $ResultPath)
        exit 0
    }

    $running = @(Get-CgrcProcesses)
    if ($running.Count -gt 0) {
        Write-Monitor ("running tier={0} processes={1}" -f $tierIndex, $running.Count)
        Start-Sleep -Seconds $CheckSeconds
        continue
    }

    $text = Read-SharedFile $RunLog
    $queueText = ""
    Get-ChildItem -LiteralPath $QueueDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "cgrc.*(err|stderr).*\.log$" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 5 |
        ForEach-Object { $queueText += "`n" + (Read-SharedFile $_.FullName) }

    if (Test-OomText ($text + "`n" + $queueText)) {
        if ($tierIndex -ge ($tiers.Count - 1)) {
            Write-Monitor "OOM detected at final tier; no lower tier remains"
            exit 2
        }
        $tierIndex += 1
        Write-Monitor ("OOM detected; downgrade to tier={0}" -f $tierIndex)
        Start-Tier $tiers[$tierIndex]
        Start-Sleep -Seconds $CheckSeconds
        continue
    }

    Write-Monitor "no CGRC process and no OOM/result detected; leaving monitor active"
    Start-Sleep -Seconds $CheckSeconds
}
