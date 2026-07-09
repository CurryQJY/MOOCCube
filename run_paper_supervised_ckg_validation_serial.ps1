param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$PythonRunner = ".\py.bat",
    [string]$StaticRunner = ".\run_usim_feedback_fast3_content_delta_static.ps1",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$OutputRootBase = "outputs\paper_supervised_ckg_validation",
    [string]$CheckpointRootBase = "checkpoints\paper_supervised_ckg_validation",
    [int[]]$SeedsToRun = @(2025, 2026, 2027),
    [string]$SeedList = "",
    [string]$DatasetList = "mooccube,junyi,coco",
    [string]$VariantList = "no_ppo_rollout,ckg_sup_t0",
    [int[]]$WaitPid = @(),
    [switch]$NoAutoWait,
    [int]$PollSeconds = 300,
    [int]$MinFreeGpuMiB = 9000,
    [switch]$SkipGpuWait,
    [int]$Epochs = 60,
    [int]$Patience = 60,
    [bool]$SaveCkpt = $true,
    [bool]$AutoResume = $true,
    [bool]$ForceFresh = $false,
    [bool]$SaveOptState = $true,
    [switch]$IgnoreLegacyResults,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

if ($SeedList.Trim().Length -gt 0) {
    $SeedsToRun = @(
        $SeedList -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { [int]$_.Trim() }
    )
}

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $repoPath $PathValue)
}

function Format-ListValue {
    param([object[]]$Value)
    return ($Value | ForEach-Object { [string]$_ }) -join ","
}

function Split-List {
    param([string]$Value)
    return @(
        $Value -split "[,\s]+" |
            Where-Object { $_.Trim().Length -gt 0 } |
            ForEach-Object { $_.Trim().ToLowerInvariant() }
    )
}

function Merge-Params {
    param(
        [hashtable]$Base,
        [hashtable]$Override
    )
    $out = @{}
    foreach ($key in $Base.Keys) {
        $out[$key] = $Base[$key]
    }
    foreach ($key in $Override.Keys) {
        $out[$key] = $Override[$key]
    }
    return $out
}

function New-Dataset {
    param(
        [string]$Name,
        [string]$Label,
        [string]$DataDir,
        [string]$RelationDir,
        [string]$PrereqGraphSource
    )
    return [ordered]@{
        Name = $Name
        Label = $Label
        DataDir = $DataDir
        RelationDir = $RelationDir
        PrereqGraphSource = $PrereqGraphSource
    }
}

function New-Variant {
    param(
        [string]$Name,
        [string]$Label,
        [string]$Rationale,
        [hashtable]$Params
    )
    return [ordered]@{
        Name = $Name
        Label = $Label
        Rationale = $Rationale
        Params = $Params
    }
}

function Get-LegacyResultRoot {
    param(
        [string]$DatasetName,
        [string]$VariantName
    )
    if ($IgnoreLegacyResults) {
        return ""
    }
    if ($DatasetName -eq "mooccube" -and $VariantName -eq "no_ppo_rollout") {
        return "outputs\content_delta_pop5\course_ppo_ablation_e60_3seed\wo_ppo_loss"
    }
    if ($DatasetName -eq "mooccube" -and $VariantName -eq "content_masked_sup") {
        return "outputs\content_delta_pop5\course_ppo_ablation_e60_3seed\static_content_masked_scorer"
    }
    return ""
}

$allDatasets = @(
    (New-Dataset `
        -Name "mooccube" `
        -Label "MOOCCube" `
        -DataDir "processed_data_hin_clean_pop5" `
        -RelationDir "MOOCCube/relations" `
        -PrereqGraphSource "concept"),
    (New-Dataset `
        -Name "junyi" `
        -Label "Junyi" `
        -DataDir "processed_data_junyi" `
        -RelationDir "processed_data_junyi\relations" `
        -PrereqGraphSource "concept"),
    (New-Dataset `
        -Name "coco" `
        -Label "COCO" `
        -DataDir "processed_data_coco" `
        -RelationDir "processed_data_coco\relations" `
        -PrereqGraphSource "concept")
)

$allVariants = @(
    (New-Variant `
        -Name "no_ppo_rollout" `
        -Label "No-PPO rollout" `
        -Rationale "Keep five-step simulator rollout and course-knowledge signals, but set PPO loss weight to zero." `
        -Params @{
            UsimSteps = 5
            PpoLossWeight = 0.0
            RolloutPolicy = "ppo"
            UseCourseFeedback = $true
            UseCourseReward = $true
            UseCourseSample = $true
            UsePrereqAux = $true
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }),
    (New-Variant `
        -Name "ckg_sup_t0" `
        -Label "CKG-Sup (T=0)" `
        -Rationale "Use supervised ranking with forced-cold masking and prerequisite auxiliary supervision; remove rollout, PPO, reward shaping, and course-aware sampling." `
        -Params @{
            UsimSteps = 0
            PpoLossWeight = 0.0
            RolloutPolicy = "ppo"
            UseCourseFeedback = $true
            UseCourseReward = $false
            UseCourseSample = $false
            UsePrereqAux = $true
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        }),
    (New-Variant `
        -Name "content_masked_sup" `
        -Label "Content-masked Sup" `
        -Rationale "Use supervised ranking with forced-cold masking, but remove simulator, PPO, course rewards, sampling, and prerequisite auxiliary supervision." `
        -Params @{
            UsimSteps = 0
            PpoLossWeight = 0.0
            RolloutPolicy = "ppo"
            UseCourseFeedback = $false
            UseCourseReward = $false
            UseCourseSample = $false
            UsePrereqAux = $false
            CourseFeedbackOnlyCold = $false
            CourseSampleOnlyCold = $false
            PrereqAuxOnlyCold = $false
        })
)

$wantedDatasets = Split-List $DatasetList
$datasets = @($allDatasets | Where-Object { $wantedDatasets -contains $_.Name })
if ($datasets.Count -lt 1) {
    throw "No datasets matched DatasetList='$DatasetList'"
}

$wantedVariants = Split-List $VariantList
$variants = @($allVariants | Where-Object { $wantedVariants -contains $_.Name })
if ($variants.Count -lt 1) {
    throw "No variants matched VariantList='$VariantList'"
}

$outputRootBaseAbs = Resolve-RepoPath $OutputRootBase
$checkpointRootBaseAbs = Resolve-RepoPath $CheckpointRootBase
$queueLog = Join-Path $outputRootBaseAbs "paper_supervised_ckg_validation_queue.log"
$planJson = Join-Path $outputRootBaseAbs "paper_supervised_ckg_validation_plan.json"
$planCsv = Join-Path $outputRootBaseAbs "paper_supervised_ckg_validation_plan.csv"

function Write-QueueLogLine {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $outputRootBaseAbs | Out-Null
        [System.IO.File]::AppendAllText($queueLog, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    Write-Host $line
}

function Get-FinalPath {
    param(
        [string]$Root,
        [int]$Seed
    )
    $tag = "strict_item_cold_balanced_thr1_seed_{0}" -f $Seed
    return Join-Path (Join-Path $Root $tag) "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
}

function Get-MissingSeeds {
    param(
        [string]$Root,
        [int[]]$Seeds
    )
    $missing = @()
    foreach ($seed in $Seeds) {
        if (-not (Test-Path -LiteralPath (Get-FinalPath -Root $Root -Seed $seed))) {
            $missing += $seed
        }
    }
    return $missing
}

function Copy-LegacySeedDirs {
    param(
        [string]$LegacyRoot,
        [string]$OutputRoot,
        [int[]]$Seeds
    )
    if ([string]::IsNullOrWhiteSpace($LegacyRoot)) {
        return
    }
    $legacyRootAbs = Resolve-RepoPath $LegacyRoot
    if (-not (Test-Path -LiteralPath $legacyRootAbs)) {
        Write-QueueLogLine "Legacy root not found; will run requested seeds instead: $LegacyRoot"
        return
    }
    New-Item -ItemType Directory -Force -Path (Resolve-RepoPath $OutputRoot) | Out-Null
    foreach ($seed in $Seeds) {
        $tag = "strict_item_cold_balanced_thr1_seed_{0}" -f $seed
        $srcDir = Join-Path $legacyRootAbs $tag
        $dstDir = Join-Path (Resolve-RepoPath $OutputRoot) $tag
        $srcFinal = Join-Path $srcDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        $dstFinal = Join-Path $dstDir "final_fullrank_usim_feedback_fast3_content_delta_static.csv"
        if (-not (Test-Path -LiteralPath $srcFinal) -or (Test-Path -LiteralPath $dstFinal)) {
            continue
        }
        if (Test-Path -LiteralPath $dstDir) {
            Write-QueueLogLine "Legacy result exists but destination is partial; not copying over existing directory: $dstDir"
            continue
        }
        Copy-Item -LiteralPath $srcDir -Destination $dstDir -Recurse
        Write-QueueLogLine "REUSE legacy result seed=$seed | src=$srcDir | dst=$dstDir"
    }
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

function Wait-ForGpu {
    param([string]$Reason)
    if ($SkipGpuWait) {
        Write-QueueLogLine "SkipGpuWait requested before $Reason."
        return
    }
    while ($true) {
        $free = Get-GpuFreeMiB
        if ($null -eq $free) {
            Write-QueueLogLine "GPU free memory unavailable before $Reason; wait ${PollSeconds}s."
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($free -ge $MinFreeGpuMiB) {
            Write-QueueLogLine "GPU free ${free}MiB >= ${MinFreeGpuMiB}MiB before $Reason."
            return
        }
        Write-QueueLogLine "GPU free ${free}MiB < ${MinFreeGpuMiB}MiB before $Reason; wait ${PollSeconds}s."
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-ActiveFast3Pids {
    $patterns = @(
        "usim_feedback_fast3_content_delta.py",
        "run_usim_feedback_fast3_content_delta_static.ps1"
    )
    $processNames = @("python.exe", "pythonw.exe", "cmd.exe", "powershell.exe", "pwsh.exe")
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $_.ProcessId -ne $PID -and
        ($processNames -contains $_.Name) -and
        $cmd -and
        ($cmd -like "*$repoPath*" -or $cmd -like "*MOOCCube*") -and
        @(($patterns | Where-Object { $cmd -like "*$_*" })).Count -gt 0
    }
    return @($procs | Select-Object -ExpandProperty ProcessId -Unique)
}

function Wait-ForPids {
    param([int[]]$Pids)
    $unique = @($Pids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
    if ($unique.Count -lt 1) {
        Write-QueueLogLine "No active experiment PID to wait for; starting supervised-CKG validation queue."
        return
    }
    Write-QueueLogLine ("Waiting for experiment PIDs before supervised-CKG validation queue: {0}" -f (Format-ListValue $unique))
    while ($true) {
        $alive = @()
        foreach ($pidValue in $unique) {
            if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
                $alive += $pidValue
            }
        }
        if ($alive.Count -lt 1) {
            Write-QueueLogLine "Waited experiment PIDs finished."
            return
        }
        Write-QueueLogLine ("Still running: {0}; wait {1}s." -f (Format-ListValue $alive), $PollSeconds)
        Start-Sleep -Seconds $PollSeconds
    }
}

$baseRunnerParams = @{
    PythonRunner = $PythonRunner
    ScriptPath = $ScriptPath
    Protocol = "strict_item_cold_balanced"
    ColdThresholds = @(1)
    Epochs = $Epochs
    Patience = $Patience
    EarlyStopAverageMode = "item_macro"
    EarlyStopScoreMode = "cold_only"
    UseContentDelta = $false
    UsePseudoColdTrain = $false
    UsePaac = $false
    CoursePrereqW = 0.08
    CoursePrereqGate = 0.20
    CourseConceptW = 0.04
    CourseDiffW = 0.03
    CourseRedundantW = 0.02
    CourseRedundantConceptGate = 1.0
    CourseRedundantMode = "concept"
    CourseTermNorm = "none"
    CourseSampleBeta = 0.20
    TrainForceCold = $true
    UseSageLite = $false
    SageUseTwoExpert = $false
    SageTwoExpertScoreFusion = $false
    UseSageAuxLoss = $false
    UseCgrcRecon = $false
    UseCourseRerank = $false
    UseStructuredHardNeg = $false
    MaskKnownPosNeg = $false
    MaskSameItemNeg = $false
    RunSampledEval = $false
    SaveCkpt = $SaveCkpt
    AutoResume = $AutoResume
    ForceFresh = $ForceFresh
    SaveOptState = $SaveOptState
}

function Build-PlanRows {
    $rows = @()
    foreach ($dataset in $datasets) {
        foreach ($variant in $variants) {
            $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
            $rows += [pscustomobject]@{
                dataset = $dataset.Name
                label = $dataset.Label
                variant = $variant.Name
                variant_label = $variant.Label
                rationale = $variant.Rationale
                seeds = Format-ListValue $SeedsToRun
                data_dir = $dataset.DataDir
                relation_dir = $dataset.RelationDir
                prereq_graph_source = $dataset.PrereqGraphSource
                train_force_cold = $params.TrainForceCold
                usim_steps = $params.UsimSteps
                ppo_loss_weight = $params.PpoLossWeight
                rollout_policy = $params.RolloutPolicy
                use_course_feedback = $params.UseCourseFeedback
                use_course_reward = $params.UseCourseReward
                use_course_sample = $params.UseCourseSample
                use_prereq_aux = $params.UsePrereqAux
                course_feedback_only_cold = $params.CourseFeedbackOnlyCold
                course_sample_only_cold = $params.CourseSampleOnlyCold
                prereq_aux_only_cold = $params.PrereqAuxOnlyCold
                mask_known_pos_neg = $params.MaskKnownPosNeg
                mask_same_item_neg = $params.MaskSameItemNeg
                legacy_result_root = Get-LegacyResultRoot -DatasetName $dataset.Name -VariantName $variant.Name
            }
        }
    }
    return $rows
}

function Write-PlanFiles {
    if ($DryRun) {
        return
    }
    New-Item -ItemType Directory -Force -Path $outputRootBaseAbs | Out-Null
    $rows = Build-PlanRows
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $planJson -Encoding UTF8
    $rows | Export-Csv -LiteralPath $planCsv -NoTypeInformation -Encoding UTF8
}

function Show-DryRunVariant {
    param(
        [object]$Dataset,
        [object]$Variant,
        [hashtable]$Params,
        [string]$OutRoot,
        [string]$CkptRoot
    )
    Write-Host ("Dataset={0}" -f $Dataset.Name)
    Write-Host ("DatasetLabel={0}" -f $Dataset.Label)
    Write-Host ("DataDir={0}" -f $Dataset.DataDir)
    Write-Host ("RelationDir={0}" -f $Dataset.RelationDir)
    Write-Host ("PrereqGraphSource={0}" -f $Dataset.PrereqGraphSource)
    Write-Host ("Variant={0}" -f $Variant.Name)
    Write-Host ("Label={0}" -f $Variant.Label)
    Write-Host ("OutputRoot={0}" -f $OutRoot)
    Write-Host ("CheckpointRoot={0}" -f $CkptRoot)
    $legacyRoot = Get-LegacyResultRoot -DatasetName $Dataset.Name -VariantName $Variant.Name
    Write-Host ("LegacyResultRoot={0}" -f $legacyRoot)
    Write-Host ("RunMode={0}" -f $(if ([string]::IsNullOrWhiteSpace($legacyRoot)) { "run" } else { "reuse_legacy_if_available" }))
    foreach ($key in @(
        "TrainForceCold",
        "UsimSteps",
        "PpoLossWeight",
        "RolloutPolicy",
        "UseCourseFeedback",
        "UseCourseReward",
        "UseCourseSample",
        "UsePrereqAux",
        "CourseFeedbackOnlyCold",
        "CourseSampleOnlyCold",
        "PrereqAuxOnlyCold",
        "MaskKnownPosNeg",
        "MaskSameItemNeg"
    )) {
        Write-Host ("{0}={1}" -f $key, $Params[$key])
    }
}

Write-PlanFiles

if ($DryRun) {
    Write-Host "Paper supervised CKG validation dry run"
    Write-Host ("Repo={0}" -f $repoPath)
    Write-Host ("StaticRunner={0}" -f $StaticRunner)
    Write-Host ("Datasets={0}" -f (Format-ListValue ($datasets.Name)))
    Write-Host ("Variants={0}" -f (Format-ListValue ($variants.Name)))
    Write-Host ("Seeds={0}" -f (Format-ListValue $SeedsToRun))
    Write-Host ("WaitPid={0}" -f (Format-ListValue $WaitPid))
    Write-Host ("AutoWait={0}" -f (-not [bool]$NoAutoWait))
    Write-Host ("Epochs={0}" -f $Epochs)
    Write-Host ("Patience={0}" -f $Patience)
    Write-Host ("IgnoreLegacyResults={0}" -f ([bool]$IgnoreLegacyResults))
    foreach ($dataset in $datasets) {
        foreach ($variant in $variants) {
            $params = Merge-Params -Base $baseRunnerParams -Override $variant.Params
            $params["DataDir"] = $dataset.DataDir
            $params["RelationDir"] = $dataset.RelationDir
            $params["PrereqGraphSource"] = $dataset.PrereqGraphSource
            $outRoot = Join-Path (Join-Path $OutputRootBase $dataset.Name) $variant.Name
            $ckptRoot = Join-Path (Join-Path $CheckpointRootBase $dataset.Name) $variant.Name
            Show-DryRunVariant -Dataset $dataset -Variant $variant -Params $params -OutRoot $outRoot -CkptRoot $ckptRoot
        }
    }
    return
}

New-Item -ItemType Directory -Force -Path $outputRootBaseAbs | Out-Null
New-Item -ItemType Directory -Force -Path $checkpointRootBaseAbs | Out-Null

Write-QueueLogLine "CONFIG supervised-CKG validation queue | datasets=$(Format-ListValue ($datasets.Name)) | variants=$(Format-ListValue ($variants.Name)) | seeds=$(Format-ListValue $SeedsToRun) | output=$OutputRootBase"
Write-QueueLogLine "PLAN files: $planJson ; $planCsv"

$waitTargets = @($WaitPid)
if (-not $NoAutoWait) {
    $autoPids = @(Get-ActiveFast3Pids)
    if ($autoPids.Count -gt 0) {
        Write-QueueLogLine ("Auto-detected active FAST3 PIDs: {0}" -f (Format-ListValue $autoPids))
        $waitTargets += $autoPids
    }
}
Wait-ForPids -Pids $waitTargets

foreach ($dataset in $datasets) {
    foreach ($variant in $variants) {
        $outRoot = Join-Path (Join-Path $OutputRootBase $dataset.Name) $variant.Name
        $ckptRoot = Join-Path (Join-Path $CheckpointRootBase $dataset.Name) $variant.Name
        $legacyRoot = Get-LegacyResultRoot -DatasetName $dataset.Name -VariantName $variant.Name
        Copy-LegacySeedDirs -LegacyRoot $legacyRoot -OutputRoot $outRoot -Seeds $SeedsToRun
        $outRootAbs = Resolve-RepoPath $outRoot
        $missingSeeds = @(Get-MissingSeeds -Root $outRootAbs -Seeds $SeedsToRun)
        if ($missingSeeds.Count -lt 1) {
            Write-QueueLogLine "SKIP $($dataset.Name)/$($variant.Name); all requested finals already exist. Aggregating."
            & $PythonRunner "aggregate_fast3_static_results.py" --root $outRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Aggregation failed for $($dataset.Name)/$($variant.Name)"
            }
            continue
        }

        Wait-ForGpu -Reason "$($dataset.Name)/$($variant.Name)"
        $runnerParams = Merge-Params -Base $baseRunnerParams -Override $variant.Params
        $runnerParams["DataDir"] = $dataset.DataDir
        $runnerParams["RelationDir"] = $dataset.RelationDir
        $runnerParams["PrereqGraphSource"] = $dataset.PrereqGraphSource
        $runnerParams["OutputRoot"] = $outRoot
        $runnerParams["CheckpointRoot"] = $ckptRoot
        $runnerParams["Seeds"] = $missingSeeds

        Write-QueueLogLine ("START {0}/{1} | label={2} | missing_seeds={3} | steps={4} | ppo_loss={5}" -f $dataset.Name, $variant.Name, $variant.Label, (Format-ListValue $missingSeeds), $runnerParams.UsimSteps, $runnerParams.PpoLossWeight)
        & $StaticRunner @runnerParams
        $exitCode = $LASTEXITCODE
        Write-QueueLogLine ("END {0}/{1} | exit_code={2}" -f $dataset.Name, $variant.Name, $exitCode)
        if ($exitCode -ne 0) {
            throw "Variant failed: $($dataset.Name)/$($variant.Name) exit_code=$exitCode"
        }
    }
}

Write-QueueLogLine "All requested supervised-CKG validation runs finished."
