param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [int]$MinFreeGpuMiB = 9000,
    [int]$PollSeconds = 60,
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo

while ($true) {
    $raw = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
    $freeMiB = 0
    if ($LASTEXITCODE -eq 0 -and $raw) {
        $first = @($raw)[0]
        if ($first -match "(\d+)") {
            $freeMiB = [int]$matches[1]
        }
    }
    $competing = @(
        Get-CimInstance Win32_Process |
            Where-Object { $_.CommandLine -match "cgrc_paper_static_hin\.py" }
    )
    $waitPidAlive = $false
    if ($WaitPid -gt 0) {
        $waitPidAlive = $null -ne (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue)
    }
    Write-Output "[$(Get-Date -Format o)] WAIT free_gpu_mib=$freeMiB competing_cgrc=$($competing.Count) wait_pid=$WaitPid wait_pid_alive=$waitPidAlive threshold=$MinFreeGpuMiB"
    if ($freeMiB -ge $MinFreeGpuMiB -and $competing.Count -eq 0 -and -not $waitPidAlive) {
        break
    }
    Start-Sleep -Seconds ([math]::Max(10, $PollSeconds))
}

Write-Output "[$(Get-Date -Format o)] START simulator factorial completion"
& (Join-Path $Repo "run_simulator_factorial_completion.ps1") -Repo $Repo
if ($LASTEXITCODE -ne 0) {
    throw "Simulator factorial completion failed: exit=$LASTEXITCODE"
}
Write-Output "[$(Get-Date -Format o)] DONE simulator factorial completion"
