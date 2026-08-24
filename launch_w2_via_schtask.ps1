# One-shot: register a short-lived scheduled task that runs the robust 3-seed
# launcher outside any agent Job Object, then start it immediately.
$ErrorActionPreference = "Stop"
$Root = "D:\DeskTop\MOOCCube"
$TaskName = "MOOCCube_prereq_w2_3seed_robust"
$Launcher = Join-Path $Root "run_w2_3seed_robust.ps1"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$HostLogDir = Join-Path $Root "background_logs"
New-Item -ItemType Directory -Path $HostLogDir -Force | Out-Null
$HostOut = Join-Path $HostLogDir "prereq_w2_task_host_$Stamp.out"
$HostErr = Join-Path $HostLogDir "prereq_w2_task_host_$Stamp.err"
$Marker = Join-Path $HostLogDir "prereq_w2_task_launch_$Stamp.txt"

if (-not (Test-Path $Launcher)) { throw "Missing $Launcher" }

# Remove previous task if present
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Launcher`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Root
# Highest available without admin: Interactive token of current user
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName

$info = Get-ScheduledTaskInfo -TaskName $TaskName
@(
  "stamp=$Stamp"
  "task=$TaskName"
  "launcher=$Launcher"
  "lastResult=$($info.LastTaskResult)"
  "lastRun=$($info.LastRunTime)"
  "state=$((Get-ScheduledTask -TaskName $TaskName).State)"
) | Set-Content -Path $Marker -Encoding UTF8

Write-Output "TASK_STARTED name=$TaskName marker=$Marker"
Write-Output "state=$((Get-ScheduledTask -TaskName $TaskName).State)"
