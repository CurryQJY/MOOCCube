param(
    [string]$Repo = "D:\DeskTop\MOOCCube",
    [string]$CampaignId = "",
    [string]$Device = "cuda:0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $repoPath

if ([string]::IsNullOrWhiteSpace($CampaignId)) {
    $CampaignId = Get-Date -Format "yyyyMMdd_HHmmss"
}
if ($CampaignId -notmatch "^[A-Za-z0-9_-]+$") {
    throw "CampaignId may contain only letters, digits, underscores, and hyphens."
}

$pythonPath = "D:\Anaconda3\envs\zw\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python runtime is missing: $pythonPath"
}

$outputRoot = "outputs\ckg_stageb_component_campaign_$CampaignId"
$checkpointRoot = "checkpoints\ckg_stageb_component_campaign_$CampaignId"
$campaignLogRoot = "background_logs\ckg_stageb_component_campaign_$CampaignId"
$scriptPath = "ckg_stageb_component_campaign.py"
$arguments = @(
    $scriptPath,
    "--campaign-id", $CampaignId,
    "--output-root", $outputRoot,
    "--checkpoint-root", $checkpointRoot,
    "--log-root", $campaignLogRoot
)
if (-not [string]::IsNullOrWhiteSpace($Device)) {
    $arguments += @("--device", $Device)
}

if ($DryRun) {
    & $pythonPath @arguments "--dry-run"
    exit $LASTEXITCODE
}

$formalRoots = @($outputRoot, $checkpointRoot)
$existingRoots = @($formalRoots | Where-Object { Test-Path -LiteralPath $_ })
if ($existingRoots.Count -gt 0) {
    throw "Component campaign requires fresh roots; refusing to reuse: $($existingRoots -join ', ')"
}
if (Test-Path -LiteralPath $campaignLogRoot) {
    throw "Component campaign log root must be fresh: $campaignLogRoot"
}
New-Item -ItemType Directory -Force -Path $campaignLogRoot | Out-Null
$stdoutPath = Join-Path $campaignLogRoot "training.out.log"
$stderrPath = Join-Path $campaignLogRoot "training.err.log"
$launchPath = Join-Path $campaignLogRoot "launch.json"
foreach ($path in @($stdoutPath, $stderrPath, $launchPath)) {
    if (Test-Path -LiteralPath $path) {
        throw "Campaign log path already exists: $path"
    }
}

$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $repoPath `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$launch = [ordered]@{
    status = "launched"
    campaign_id = $CampaignId
    pid = $process.Id
    repo = $repoPath
    output_root = (Join-Path $repoPath $outputRoot)
    checkpoint_root = (Join-Path $repoPath $checkpointRoot)
    campaign_log_root = (Join-Path $repoPath $campaignLogRoot)
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    test_evaluation = $false
    component = "soft_anchor_l2"
    started_at = (Get-Date).ToString("o")
}
$launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding UTF8
$launch | ConvertTo-Json -Depth 8
