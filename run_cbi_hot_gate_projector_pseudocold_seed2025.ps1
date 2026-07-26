param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath
$outputRoot = "outputs\cbi_hot_gate_projector_pseudocold_seed2025"
$checkpointRoot = "checkpoints\cbi_hot_gate_projector_pseudocold_seed2025"
$logRoot = "background_logs\cbi_hot_gate_projector_pseudocold_seed2025"
$manifestPath = Join-Path $outputRoot "run_manifest.json"
$logPath = Join-Path $logRoot "training.log"

$runnerParams = [ordered]@{
    PythonRunner=".\py.bat"; ScriptPath="cbi_hot_gate_projector_pseudocold_seed2025.py"
    DataDir="processed_data_hin_clean_pop5"; RelationDir="MOOCCube/relations"
    OutputRoot=$outputRoot; CheckpointRoot=$checkpointRoot; Protocol="strict_item_cold_balanced"
    ColdThresholds=@(1); Seeds=@(2025); Epochs=8; Patience=8
    EarlyStopAverageMode="item_macro"; EarlyStopScoreMode="balanced_rn"
    UseContentDelta=$true; ContentDeltaPaperStyle=$true; ContentDeltaReplaceItem=$false
    ContentDeltaColdOnly=$true; ContentDeltaTrainOnIdDropout=$false
    ContentDeltaMode="projector"; ContentDeltaMaxNorm=0.20; ContentDeltaScale=1.0
    ContentDeltaLrMult=1.0; ContentDeltaL2W=0.0; ContentDeltaCapW=0.0
    ContentDeltaAuxMode="base"; AuxWeight=0.3; AuxHotOnly=$false
    UsePseudoColdTrain=$true; PseudoColdMode="batch_random"; PseudoColdRatio=0.30; PseudoColdMinPop=5
    UsePaac=$false; UseCourseFeedback=$true; UseCourseReward=$true; UseCourseSample=$true
    UsePrereqAux=$true; PrereqGraphSource="concept"; CoursePrereqW=0.08; CourseConceptW=0.04
    CourseDiffW=0.03; CourseRedundantW=0.02; CourseRedundantMode="concept"; CourseTermNorm="none"
    CourseFeedbackOnlyCold=$false; CourseSampleOnlyCold=$false; PrereqAuxOnlyCold=$false; CourseSampleBeta=0.20
    UseSageLite=$false; SageTwoExpertScoreFusion=$false; UseSageAuxLoss=$false; UseCourseRerank=$false
    UseStructuredHardNeg=$false; MaskKnownPosNeg=$true; MaskSameItemNeg=$true; TrainForceCold=$true
    UsimSteps=5; UseUsimRefinedEval=$true; PpoLossWeight=1.0; RolloutPolicy="ppo"; RunSampledEval=$false
    SaveCkpt=$true; AutoResume=$false; ForceFresh=$true; SaveOptState=$true
}
$lockedConfig = [ordered]@{
    experiment="cbi_hot_gate_projector_pseudocold_seed2025"
    method="hot_gate_native_scale_shared_projector_pseudocold"
    hot_only_gate=$true; cold_bypasses_gate=$true; normalize_hot_fused_before_simulation=$false
    content_delta_mode="projector"; content_delta_cold_only=$true; content_delta_max_norm=0.20
    pseudo_cold_train=$true; pseudo_cold_mode="batch_random"; pseudo_cold_ratio=0.30
    target_anchor="initial_cbi"; selection="balanced_rn_with_epoch_snapshots"
    snapshot_epochs=@(1,2,3,4,5,6,7,8); runner_parameters=$runnerParams
}
if ($DryRun) { $lockedConfig | ConvertTo-Json -Depth 20; exit 0 }
function Get-HashMap([string[]]$Paths) { $r=[ordered]@{}; foreach($p in $Paths){if(-not(Test-Path $p)){throw "Missing file: $p"};$r[$p]=(Get-FileHash $p -Algorithm SHA256).Hash.ToLower()};return $r }
function Write-Json([string]$Path,$Value){$parent=Split-Path -Parent $Path;New-Item -ItemType Directory -Force $parent|Out-Null;$Value|ConvertTo-Json -Depth 30|Set-Content $Path -Encoding UTF8}
$sourceFiles=@("run_cbi_hot_gate_projector_pseudocold_seed2025.ps1","cbi_hot_gate_projector_pseudocold_seed2025.py","cbi_hot_gate_audit_seed2025.py","cbi_anchor_sim.py","cbi_trust_sim.py","evaluate_cbi_all_refined_seed2025.py","run_usim_feedback_fast3_content_delta_static.ps1")
$protectedFiles=@("usim_feedback_fast3_content_delta.py","fast3_delta\eval.py","fast3_delta\config.py","run_fast3_main_table_config.ps1","paper_aaai27\main.tex")
New-Item -ItemType Directory -Force $outputRoot,$checkpointRoot,$logRoot|Out-Null
if(Test-Path $manifestPath){throw "Existing manifest found: $manifestPath"}
$before=Get-HashMap $protectedFiles
$manifest=[ordered]@{schema_version=1;experiment="cbi_hot_gate_projector_pseudocold_seed2025";status="running";started_at_utc=(Get-Date).ToUniversalTime().ToString("o");completed_at_utc=$null;elapsed_seconds=$null;exit_code=$null;error=$null;repo=$repoPath;git_commit=(git rev-parse HEAD).Trim();git_dirty=@((git status --porcelain)).Count -gt 0;locked_config=$lockedConfig;source_sha256=Get-HashMap $sourceFiles;protected_files_before=$before;protected_files_after=$null}
Write-Json $manifestPath $manifest
$timer=[Diagnostics.Stopwatch]::StartNew();$runError=$null;$oldSnapshot=$env:USIM_FB_SNAPSHOT_EPOCHS
try{$env:USIM_FB_SNAPSHOT_EPOCHS="1,2,3,4,5,6,7,8";& ".\run_usim_feedback_fast3_content_delta_static.ps1" @runnerParams *>&1|Tee-Object -FilePath $logPath -Append;if(-not $?){throw "Static runner returned unsuccessful status."};$manifest.status="completed";$manifest.exit_code=0}
catch{$runError=$_;$manifest.status="failed";$manifest.exit_code=1;$manifest.error=$_.Exception.Message}
finally{if($null -eq $oldSnapshot){Remove-Item Env:USIM_FB_SNAPSHOT_EPOCHS -ErrorAction SilentlyContinue}else{$env:USIM_FB_SNAPSHOT_EPOCHS=$oldSnapshot};$timer.Stop();$after=Get-HashMap $protectedFiles;$manifest.protected_files_after=$after;$changed=@($protectedFiles|Where-Object{$before[$_] -ne $after[$_]});if($changed.Count -gt 0){$manifest.status="failed";$manifest.exit_code=1;$manifest.error="Protected files changed: $($changed -join ', ')";$runError=[InvalidOperationException]::new($manifest.error)};$manifest.completed_at_utc=(Get-Date).ToUniversalTime().ToString("o");$manifest.elapsed_seconds=[Math]::Round($timer.Elapsed.TotalSeconds,3);Write-Json $manifestPath $manifest}
if($null -ne $runError){throw $runError};Write-Host "Hot-gate projector pseudo-cold seed-2025 experiment completed."
