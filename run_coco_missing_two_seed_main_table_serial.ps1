param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$DataDir = "processed_data_coco",
    [string]$RelationDir = "processed_data_coco\relations",
    [string]$OutputRoot = "outputs\coco\single_seed_triage",
    [string]$CheckpointRoot = "checkpoints\coco\single_seed_triage",
    [int[]]$Seeds = @(2026, 2027),
    [int]$ColdThreshold = 1,
    [int]$EvalNeg = 200,
    [bool]$RunOurs = $true,
    [int]$OursEpochs = 30,
    [int]$OursPatience = 10,
    [int]$OursBatchSize = 2048,
    [bool]$RunLightweightBaselines = $true,
    [string[]]$LightweightModels = @("Popularity", "ContentProfile", "BPR", "LightGCN", "DropoutNet", "GAR", "CCFCRec", "LightGCL"),
    [int]$BaselineEpochs = 10,
    [int]$BaselineBatchSize = 1024,
    [switch]$UseGpuBaselines,
    [switch]$CpuBaselines,
    [bool]$RunALDI = $true,
    [int]$ALDITeacherEpochs = 100,
    [int]$ALDIStudentEpochs = 100,
    [int]$ALDIEvalInterval = 10,
    [int]$ALDIBatchSize = 1024,
    [switch]$IncludeCgrc,
    [int]$CgrcEpochs = 50,
    [int]$CgrcBatchSize = 512,
    [int]$CgrcReconUserChunk = 256,
    [int]$CgrcReconTopK = 20,
    [int]$PollSeconds = 120,
    [switch]$AllowConcurrent,
    [switch]$RerunCompleted,
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

function Join-ModelNames([string[]]$Names) {
    $out = @()
    foreach ($entry in $Names) {
        foreach ($part in ([string]$entry -split ",")) {
            $name = $part.Trim()
            if ($name) {
                $out += $name
            }
        }
    }
    return $out
}

$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location $Repo

$OutputRootAbs = Resolve-RunPath $Repo $OutputRoot
$CheckpointRootAbs = Resolve-RunPath $Repo $CheckpointRoot
$QueueDir = Join-Path $OutputRootAbs "_queue"
$QueueLog = Join-Path $QueueDir "missing_two_seed_main_table_serial.log"
$OursRunner = Join-Path $Repo "run_xes3g5m_ours_sota_serial.ps1"
$BaselineRunner = Join-Path $Repo "run_xes3g5m_lightweight_baselines.ps1"
$CgrcRunner = Join-Path $Repo "run_coco_cgrc_paper_single_seed.ps1"

if ($UseGpuBaselines -and $CpuBaselines) {
    throw "Use only one of -UseGpuBaselines or -CpuBaselines."
}
$EffectiveUseGpuBaselines = (-not $CpuBaselines.IsPresent) -or $UseGpuBaselines.IsPresent

New-Item -ItemType Directory -Force -Path $QueueDir, $OutputRootAbs, $CheckpointRootAbs | Out-Null

function Write-RunLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        Add-Content -LiteralPath $QueueLog -Encoding UTF8 -Value $line
    }
    Write-Host $line
}

function Split-Name([int]$Seed) {
    "strict_item_cold_balanced_thr{0}_seed_{1}" -f $ColdThreshold, $Seed
}

function Ours-SplitDir([int]$Seed) {
    Join-Path (Join-Path $OutputRootAbs "ours_full") (Split-Name $Seed)
}

function Test-OursFinished([int]$Seed) {
    Test-Path -LiteralPath (Join-Path (Ours-SplitDir $Seed) "final_fullrank_usim_feedback_fast3_content_delta_static.csv")
}

function Test-BaselineFinished([int]$Seed, [string]$ResultFile) {
    Test-Path -LiteralPath (Join-Path (Join-Path (Ours-SplitDir $Seed) "main_table_compare") $ResultFile)
}

function Assert-Inputs {
    foreach ($path in @($DataDir, $RelationDir)) {
        $abs = Resolve-RunPath $Repo $path
        if (-not (Test-Path -LiteralPath $abs)) {
            throw "Missing required COCO input path: $abs"
        }
    }
    foreach ($script in @($OursRunner, $BaselineRunner)) {
        if (-not (Test-Path -LiteralPath $script)) {
            throw "Missing required runner: $script"
        }
    }
    if ($IncludeCgrc -and (-not (Test-Path -LiteralPath $CgrcRunner))) {
        throw "Missing CGRC runner: $CgrcRunner"
    }
}

function Get-ActiveCgrcProcesses {
    @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $cmd = [string]$_.CommandLine
                ($_.ProcessId -ne $PID) -and (
                    ($_.Name -eq "python.exe" -and $cmd -like "*cgrc_paper_static_hin.py*") -or
                    ($_.Name -eq "cmd.exe" -and $cmd -like "*cgrc_paper_static_hin.py*") -or
                    ($_.Name -eq "powershell.exe" -and $cmd -like "*run_coco_cgrc_paper_single_seed.ps1*") -or
                    ($_.Name -eq "powershell.exe" -and $cmd -like "*monitor_coco_cgrc_auto_downgrade.ps1*")
                )
            }
    )
}

function Wait-ForCurrentCgrc {
    if ($AllowConcurrent -or $DryRun) {
        return
    }
    while ($true) {
        $running = Get-ActiveCgrcProcesses
        if ($running.Count -eq 0) {
            return
        }
        $ids = ($running | Select-Object -ExpandProperty ProcessId) -join ","
        Write-RunLog "WAIT existing CGRC process(es) before starting next COCO seed: pid=$ids"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    Write-RunLog "START $Name"
    if ($DryRun) {
        & $Body
        Write-RunLog "DRYRUN END $Name"
        return
    }
    & $Body
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit=$LASTEXITCODE"
    }
    Write-RunLog "END $Name"
}

function Invoke-OursSeed([int]$Seed) {
    if ((Test-OursFinished $Seed) -and (-not $RerunCompleted)) {
        Write-RunLog "SKIP Ours seed=$Seed | completed=$(Join-Path (Ours-SplitDir $Seed) 'final_fullrank_usim_feedback_fast3_content_delta_static.csv')"
        return
    }
    Wait-ForCurrentCgrc
    $params = @{
        Repo = $Repo
        PythonRunner = $PythonRunner
        DataDir = $DataDir
        RelationDir = $RelationDir
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        Seeds = @($Seed)
        Epochs = $OursEpochs
        Patience = $OursPatience
        ColdThreshold = $ColdThreshold
        EvalNeg = $EvalNeg
        OursBatchSize = $OursBatchSize
        ContentProfileBatchSize = $BaselineBatchSize
        MaskKnownPosNeg = $true
        MaskSameItemNeg = $true
        ForceFresh = $false
        AutoResume = $true
        SkipNoCourse = $true
        SkipContentProfile = $true
        SkipCgrc = $true
        SkipAggregate = $true
    }
    if ($AllowConcurrent) { $params["AllowConcurrent"] = $true }
    if ($RerunCompleted) { $params["RerunCompleted"] = $true }
    if ($DryRun) { $params["DryRun"] = $true }

    Invoke-Step "Ours seed=$Seed" {
        Write-Host ("Invoke: {0} seed={1} epochs={2} out={3}" -f $OursRunner, $Seed, $OursEpochs, (Ours-SplitDir $Seed))
        if (-not $DryRun) {
            & $OursRunner @params
        }
    }
}

function Invoke-LightweightSeed([int]$Seed, [string[]]$Models) {
    $modelNames = Join-ModelNames $Models
    if ($modelNames.Count -eq 0) {
        return
    }
    Wait-ForCurrentCgrc
    $params = @{
        Repo = $Repo
        PythonRunner = $PythonRunner
        DataDir = $DataDir
        RelationDir = $RelationDir
        PrereqGraphSource = "concept"
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        Seeds = @($Seed)
        ColdThreshold = $ColdThreshold
        EvalNeg = $EvalNeg
        Models = $modelNames
        PopBatchSize = $BaselineBatchSize
        ContentProfileBatchSize = $BaselineBatchSize
        BprEpochs = $BaselineEpochs
        BprEvalInterval = $BaselineEpochs
        BprBatchSize = 4096
        LightGCNEpochs = $BaselineEpochs
        LightGCNEvalInterval = $BaselineEpochs
        LightGCNBatchSize = 2048
        DropoutEpochs = $BaselineEpochs
        DropoutEvalInterval = $BaselineEpochs
        DropoutBatchSize = $BaselineBatchSize
        GarEpochs = $BaselineEpochs
        GarEvalInterval = $BaselineEpochs
        GarBatchSize = $BaselineBatchSize
        CCFCEpochs = $BaselineEpochs
        CCFCEvalInterval = $BaselineEpochs
        CCFCBatchSize = $BaselineBatchSize
        LightGCLEpochs = $BaselineEpochs
        LightGCLEvalInterval = $BaselineEpochs
        LightGCLBatchSize = $BaselineBatchSize
    }
    if ($EffectiveUseGpuBaselines) { $params["UseGpu"] = $true }
    if ($AllowConcurrent) { $params["AllowConcurrent"] = $true }
    if ($RerunCompleted) { $params["RerunCompleted"] = $true }
    if ($DryRun) { $params["DryRun"] = $true }

    Invoke-Step "Lightweight baselines seed=$Seed models=$($modelNames -join ',')" {
        Write-Host ("Invoke: {0} seed={1} models={2}" -f $BaselineRunner, $Seed, ($modelNames -join ","))
        if (-not $DryRun) {
            & $BaselineRunner @params
        }
    }
}

function Invoke-ALDISeed([int]$Seed) {
    if ((Test-BaselineFinished $Seed "aldi_static_result.json") -and (-not $RerunCompleted)) {
        Write-RunLog "SKIP ALDI seed=$Seed | completed=$(Join-Path (Join-Path (Ours-SplitDir $Seed) 'main_table_compare') 'aldi_static_result.json')"
        return
    }
    Wait-ForCurrentCgrc
    $params = @{
        Repo = $Repo
        PythonRunner = $PythonRunner
        DataDir = $DataDir
        RelationDir = $RelationDir
        PrereqGraphSource = "concept"
        OutputRoot = $OutputRoot
        CheckpointRoot = $CheckpointRoot
        Seeds = @($Seed)
        ColdThreshold = $ColdThreshold
        EvalNeg = $EvalNeg
        Models = @("ALDI")
        ALDITeacherEpochs = $ALDITeacherEpochs
        ALDIStudentEpochs = $ALDIStudentEpochs
        ALDIEvalInterval = $ALDIEvalInterval
        ALDIEmbDim = 64
        ALDIBatchSize = $ALDIBatchSize
    }
    if ($EffectiveUseGpuBaselines) { $params["UseGpu"] = $true }
    if ($AllowConcurrent) { $params["AllowConcurrent"] = $true }
    if ($RerunCompleted) { $params["RerunCompleted"] = $true }
    if ($DryRun) { $params["DryRun"] = $true }

    Invoke-Step "ALDI seed=$Seed teacher=$ALDITeacherEpochs student=$ALDIStudentEpochs" {
        Write-Host ("Invoke: {0} seed={1} model=ALDI teacher={2} student={3}" -f $BaselineRunner, $Seed, $ALDITeacherEpochs, $ALDIStudentEpochs)
        if (-not $DryRun) {
            & $BaselineRunner @params
        }
    }
}

function Invoke-CgrcSeed([int]$Seed) {
    if ((Test-BaselineFinished $Seed "cgrc_paper_static_result.json") -and (-not $RerunCompleted)) {
        Write-RunLog "SKIP CGRC-paper seed=$Seed | completed=$(Join-Path (Join-Path (Ours-SplitDir $Seed) 'main_table_compare') 'cgrc_paper_static_result.json')"
        return
    }
    Wait-ForCurrentCgrc
    Invoke-Step "CGRC-paper seed=$Seed epochs=$CgrcEpochs batch=$CgrcBatchSize chunk=$CgrcReconUserChunk" {
        Write-Host ("Invoke: {0} seed={1} epochs={2} batch={3} chunk={4}" -f $CgrcRunner, $Seed, $CgrcEpochs, $CgrcBatchSize, $CgrcReconUserChunk)
        if (-not $DryRun) {
            & $CgrcRunner -Seed $Seed -Epochs $CgrcEpochs -BatchSize $CgrcBatchSize -ReconUserChunk $CgrcReconUserChunk -ReconTopK $CgrcReconTopK
        }
    }
}

function Invoke-FinalAggregate {
    Invoke-Step "aggregate finished baselines" {
        $outDir = Join-Path $OutputRoot "main_table_compare"
        Write-Host ("Invoke aggregate -> {0}" -f (Resolve-RunPath $Repo $outDir))
        if (-not $DryRun) {
            & $PythonRunner .\aggregate_main_table_static_results.py `
                --root (Join-Path $OutputRoot "ours_full") `
                --split-glob "strict_item_cold_balanced_thr*_seed_*" `
                --result-subdir "main_table_compare" `
                --metric-mode "item_macro" `
                --out-dir $outDir
        }
    }
}

Assert-Inputs

Write-RunLog ("QUEUE START COCO missing two-seed main-table serial | seeds={0} | run_ours={1} | lightweight={2} | aldi={3} | cgrc={4} | gpu_baselines={5}" -f `
    ($Seeds -join ","), $RunOurs, $RunLightweightBaselines, $RunALDI, [bool]$IncludeCgrc, [bool]$EffectiveUseGpuBaselines)

foreach ($seed in $Seeds) {
    Write-RunLog "SEED START $seed"
    if ($RunOurs) {
        Invoke-OursSeed $seed
    }
    if ($RunLightweightBaselines) {
        Invoke-LightweightSeed $seed $LightweightModels
    }
    if ($RunALDI) {
        Invoke-ALDISeed $seed
    }
    if ($IncludeCgrc) {
        Invoke-CgrcSeed $seed
    }
    Write-RunLog "SEED END $seed"
}

Invoke-FinalAggregate
Write-RunLog "QUEUE DONE COCO missing two-seed main-table serial"

Write-Host ""
Write-Host "Logs:"
Write-Host ("  Queue: {0}" -f $QueueLog)
Write-Host ("  Aggregate table: {0}" -f (Join-Path $OutputRootAbs "main_table_compare\main_table_item_macro_paper_narrow.csv"))
Write-Host ("  Detail table: {0}" -f (Join-Path $OutputRootAbs "main_table_compare\main_table_item_macro_detail.csv"))
