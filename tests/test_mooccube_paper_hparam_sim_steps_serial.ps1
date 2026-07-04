$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_mooccube_paper_hparam_sim_steps_serial.ps1"
$worker = Join-Path $repo "run_mooccube_paper_hparam_sim_steps_worker.cmd"
$launcher = Join-Path $repo "launch_mooccube_paper_hparam_sim_steps_now.cmd"
$staticRunner = Join-Path $repo "run_usim_feedback_fast3_content_delta_static.ps1"

foreach ($path in @($script, $worker, $launcher)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing simulation-step hyperparam helper: $path"
    }
}

$staticText = Get-Content -Raw -Encoding UTF8 -LiteralPath $staticRunner
foreach ($expected in @(
    "[int]`$UsimSteps = 5",
    "USIM_STEPS"
)) {
    if (-not $staticText.Contains($expected)) {
        throw "Static runner does not expose simulation-step control: $expected"
    }
}

$out = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script `
    -Repo $repo `
    -SeedList "2025" `
    -VariantList "sim_steps_1,sim_steps_3,sim_steps_7,sim_steps_10" `
    -OutputRootBase "outputs\_dryrun_paper_hparam_sim_steps" `
    -CheckpointRootBase "checkpoints\_dryrun_paper_hparam_sim_steps" `
    -WaitPid 12345 `
    -NoAutoWait `
    -DryRun

if ($LASTEXITCODE -ne 0) {
    throw "Simulation-step hyperparam dry-run exited with code $LASTEXITCODE"
}

$text = ($out -join "`n")
foreach ($expected in @(
    "MOOCCube paper-main simulation-step hyperparam dry run",
    "BaselineRoot=outputs\content_delta_pop5\course_ablation_e60_3seed\full",
    "OutputRootBase=outputs\_dryrun_paper_hparam_sim_steps",
    "CheckpointRootBase=checkpoints\_dryrun_paper_hparam_sim_steps",
    "Seeds=2025",
    "WaitPid=12345",
    "RunCount=4",
    "UseCourseFeedback=True",
    "UseCourseReward=True",
    "UseCourseSample=True",
    "UsePrereqAux=True",
    "CourseSampleBeta=0.2",
    "CoursePrereqW=0.08",
    "CourseConceptW=0.04",
    "CourseDiffW=0.03",
    "CourseRedundantW=0.02",
    "Variant=sim_steps_1",
    "UsimSteps=1",
    "Variant=sim_steps_3",
    "UsimSteps=3",
    "Variant=sim_steps_7",
    "UsimSteps=7",
    "Variant=sim_steps_10",
    "UsimSteps=10"
)) {
    if ($text -notlike "*$expected*") {
        throw "Missing expected simulation-step dry-run line: $expected"
    }
}

$workerText = Get-Content -Raw -Encoding UTF8 -LiteralPath $worker
foreach ($expected in @(
    "set `"VARIANT_LIST=sim_steps_1,sim_steps_3,sim_steps_7,sim_steps_10`"",
    "-SeedList `"2025,2026,2027`"",
    "-VariantList `"%VARIANT_LIST%`"",
    "course_hparam_sim_steps_worker_latest_paths.txt",
    "course_hparam_sim_steps_queue.log"
)) {
    if (-not $workerText.Contains($expected)) {
        throw "Simulation-step worker is missing expected configuration: $expected"
    }
}

foreach ($unexpected in @("prereq_gate", "term_norm", "reward_scale", "sample_beta")) {
    if ($workerText.Contains($unexpected)) {
        throw "Simulation-step worker should not include unrelated hyperparam: $unexpected"
    }
}

Write-Host "test_mooccube_paper_hparam_sim_steps_serial.ps1 passed"
