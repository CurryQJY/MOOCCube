param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback_fast3_content_delta.py",
    [string]$Root = "outputs\content_delta_pop5\fullstack_course_ablation_legacy",
    [string]$CkptRoot = "checkpoints\content_delta_pop5\fullstack_course_ablation_legacy"
)

$ErrorActionPreference = "Stop"

$env:USIM_LEGACY_TRAIN_PROTOCOL = "1"

& "$PSScriptRoot\run_fullstack_course_ablation.ps1" `
    -PythonRunner $PythonRunner `
    -ScriptPath $ScriptPath `
    -Root $Root `
    -CkptRoot $CkptRoot
