param(
    [string]$Repo = "D:\DeskTop\MOOCCube"
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$scriptPath = Join-Path $repoPath "run_mooccube_paper_core_ablation_serial.ps1"
$outRoot = Join-Path $repoPath "outputs\content_delta_pop5\course_core_ablation_e60_3seed"
$wrapperLog = Join-Path $outRoot "core_ablation_task_wrapper.log"

function Write-WrapperLog {
    param([string]$Message)
    New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $wrapperLog -Encoding UTF8 -Value "[$stamp] $Message"
}

try {
    Write-WrapperLog "WRAPPER_START pid=$PID repo=$repoPath script=$scriptPath"
    & $scriptPath `
        -Repo $repoPath `
        -NoAutoWait `
        -PollSeconds 300 `
        -MinFreeGpuMiB 9000

    if (-not $?) {
        Write-WrapperLog "WRAPPER_END success=false"
        exit 1
    }

    Write-WrapperLog "WRAPPER_END success=true"
    exit 0
} catch {
    Write-WrapperLog "WRAPPER_ERROR $($_.Exception.Message)"
    if ($_.ScriptStackTrace) {
        Write-WrapperLog "WRAPPER_STACK $($_.ScriptStackTrace)"
    }
    exit 1
}
