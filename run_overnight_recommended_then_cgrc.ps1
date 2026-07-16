param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$queueRoot = Join-Path $Repo "background_logs\overnight_recommended_then_cgrc"
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
$failurePath = Join-Path $queueRoot "stage_failures.json"
$failures = [System.Collections.Generic.List[object]]::new()

function Save-Failures {
    ConvertTo-Json -InputObject @($failures) -Depth 5 | Set-Content -LiteralPath $failurePath -Encoding utf8
}

function Invoke-Stage([string]$name, [string]$scriptName) {
    $path = Join-Path $Repo $scriptName
    Write-Host "[$(Get-Date -Format o)] STAGE START name=$name script=$scriptName"
    try {
        & $path -Repo $Repo -DryRun:$DryRun
        Write-Host "[$(Get-Date -Format o)] STAGE DONE name=$name"
    }
    catch {
        $entry = [ordered]@{ stage = $name; script = $scriptName; time = (Get-Date -Format o); error = $_.Exception.Message }
        $failures.Add([pscustomobject]$entry)
        Save-Failures
        Write-Host "[$(Get-Date -Format o)] STAGE FAILED name=$name error=$($_.Exception.Message)"
    }
}

Save-Failures
Invoke-Stage "cross_dataset_actor_ab" "run_cross_dataset_actor_inference_ab.ps1"
Invoke-Stage "mooccube_policy_controls" "run_mooccube_test_policy_controls.ps1"
Invoke-Stage "wo_knowledge_sampler_3seed" "run_recovered_missing_ablation_wo_sampler.ps1"
Invoke-Stage "cgrc_p1_resume_3seed" "run_cgrc_p1_reproduction_resume.ps1"
Save-Failures
Write-Host "[$(Get-Date -Format o)] QUEUE END failures=$($failures.Count) failure_manifest=$failurePath"
if ($failures.Count -gt 0) { exit 1 }
