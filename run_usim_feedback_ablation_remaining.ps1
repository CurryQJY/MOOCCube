param(
    [string]$PythonRunner = ".\py.bat",
    [string]$ScriptPath = "usim_feedback.py",
    [string]$OutputRoot = "outputs\usim_feedback_ablation",
    [string]$CheckpointRoot = "checkpoints\usim_feedback_ablation"
)

$ErrorActionPreference = "Stop"

& ".\run_usim_feedback_ablation.ps1" `
    -PythonRunner $PythonRunner `
    -ScriptPath $ScriptPath `
    -OutputRoot $OutputRoot `
    -CheckpointRoot $CheckpointRoot `
    -SkipExperiments current
