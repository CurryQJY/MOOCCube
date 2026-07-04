param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid,
    [Parameter(Mandatory = $true)]
    [string]$Script,
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"

Write-Host ("Waiting for PID {0} to exit..." -f $WaitPid) -ForegroundColor Cyan
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

Write-Host ("PID {0} exited. Starting {1}" -f $WaitPid, $Script) -ForegroundColor Green
& $Script
