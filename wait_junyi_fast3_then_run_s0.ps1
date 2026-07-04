param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$JunyiOutputRoot = "outputs\junyi\main_table_3seed",
    [string[]]$JunyiSeeds = @("2026", "2027"),
    [int]$WaitPid = 0,
    [int]$PollSeconds = 300,
    [string]$S0Runner = ".\run_fast3_main_table_config.ps1",
    [string]$S0OutputRoot = "outputs\content_delta_pop5\sage_lite_v1\S0_sage",
    [string]$S0CheckpointRoot = "checkpoints\content_delta_pop5\sage_lite_v1\S0_sage",
    [int]$S0Seed = 2025,
    [int]$SagePoolTopK = 48,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Resolve-RunPath([string]$Base, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $Base $Path)
}

function Parse-SeedList([string[]]$RawSeeds) {
    $parsed = @()
    foreach ($rawSeed in $RawSeeds) {
        foreach ($part in ([string]$rawSeed -split ",")) {
            $trimmed = $part.Trim()
            if ($trimmed) {
                $parsed += [int]$trimmed
            }
        }
    }
    return [int[]]$parsed
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$JunyiSeeds = Parse-SeedList $JunyiSeeds
$JunyiOutputRootAbs = Resolve-RunPath $Repo $JunyiOutputRoot
$S0OutputRootAbs = Resolve-RunPath $Repo $S0OutputRoot
$S0CheckpointRootAbs = Resolve-RunPath $Repo $S0CheckpointRoot
$S0RunnerAbs = Resolve-RunPath $Repo $S0Runner

$QueueDir = Join-Path $S0OutputRootAbs "_watch_junyi_fast3_then_s0"
$QueueLog = Join-Path $QueueDir "queue.log"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

function Write-QueueLogLine([string]$Path, [string]$Line) {
    $payload = $Line + [Environment]::NewLine
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            [System.IO.File]::AppendAllText($Path, $payload, [System.Text.Encoding]::UTF8)
            return
        } catch {
            if ($attempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds (200 * $attempt)
        }
    }
}

function Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-QueueLogLine $QueueLog $line
    Write-Host $line
}

function Get-JunyiFast3FinalPath([int]$Seed) {
    $splitDir = Join-Path $JunyiOutputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $Seed)
    return Join-Path $splitDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Test-JunyiFast3Done {
    $allDone = $true
    foreach ($seed in $JunyiSeeds) {
        $finalPath = Get-JunyiFast3FinalPath $seed
        $exists = (Test-Path -LiteralPath $finalPath)
        if ($exists) {
            $item = Get-Item -LiteralPath $finalPath
            $exists = $item.Length -gt 0
        }
        if ($exists) {
            Log "JUNYI FAST3 DONE seed=$seed | final=$finalPath"
        } else {
            Log "JUNYI FAST3 WAIT seed=$seed | missing=$finalPath"
            $allDone = $false
        }
    }
    return $allDone
}

function Wait-JunyiFast3 {
    Log "WATCH START Junyi FAST3 seeds=$($JunyiSeeds -join ',') | root=$JunyiOutputRootAbs"

    if ($DryRun) {
        if (-not (Test-JunyiFast3Done)) {
            throw "DryRun requires all Junyi FAST3 final files to exist."
        }
        return
    }

    if ($WaitPid -gt 0) {
        Log "WAIT PID=$WaitPid before file polling"
        while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
            Start-Sleep -Seconds $PollSeconds
        }
        Log "PID exited | pid=$WaitPid"
    }

    while (-not (Test-JunyiFast3Done)) {
        Start-Sleep -Seconds $PollSeconds
    }
}

function Start-S0 {
    $s0Final = Join-Path (
        Join-Path $S0OutputRootAbs ("strict_item_cold_balanced_thr1_seed_{0}" -f $S0Seed)
    ) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"

    if ((-not $DryRun) -and (Test-Path -LiteralPath $s0Final)) {
        Log "SKIP S0 seed=$S0Seed | exists=$s0Final"
        return
    }

    if ($DryRun) {
        Log "DRYRUN START S0 seed=$S0Seed | out=$S0OutputRootAbs | ckpt=$S0CheckpointRootAbs | UseSageLite=True | SagePoolTopK=$SagePoolTopK"
        return
    }

    Log "START S0 seed=$S0Seed | out=$S0OutputRootAbs | ckpt=$S0CheckpointRootAbs | runner=$S0RunnerAbs"
    & $S0RunnerAbs `
        -Repo $Repo `
        -OutputRoot $S0OutputRootAbs `
        -CheckpointRoot $S0CheckpointRootAbs `
        -Seeds $S0Seed `
        -UseSageLite `
        -SagePoolTopK $SagePoolTopK
    if ($LASTEXITCODE -ne 0) {
        throw "S0 run failed with exit code $LASTEXITCODE"
    }
    Log "DONE S0 seed=$S0Seed | out=$S0OutputRootAbs"
}

Wait-JunyiFast3
Start-S0
Log "WATCH DONE Junyi FAST3 -> S0"
