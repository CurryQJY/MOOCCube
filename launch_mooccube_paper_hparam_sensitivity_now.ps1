$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$outDir = Join-Path $repo "outputs\content_delta_pop5\course_hparam_sensitivity_e60_3seed"
$worker = Join-Path $repo "run_mooccube_paper_hparam_sensitivity_worker.cmd"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $worker
$psi.Arguments = ""
$psi.WorkingDirectory = $repo
$psi.UseShellExecute = $true
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden

$process = [System.Diagnostics.Process]::Start($psi)
Write-Host ("PID={0}" -f $process.Id)
Write-Host ("PATHS={0}" -f (Join-Path $outDir "course_hparam_sensitivity_worker_latest_paths.txt"))
