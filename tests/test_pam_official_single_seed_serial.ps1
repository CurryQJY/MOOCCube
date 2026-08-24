$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$script = Join-Path $repo "run_pam_official_single_seed_serial.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing PAM official serial runner: $script"
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pam_official_serial_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null

try {
    $out = & $script `
        -OutputRoot (Join-Path $tmpRoot "outputs") `
        -DryRun *>&1
    $text = ($out -join "`n")

    foreach ($needle in @(
        "Total tasks: 3",
        "dataset=mooccube seed=2025",
        "dataset=junyi seed=2025",
        "dataset=coco seed=2025",
        "processed_data_hin_clean_pop5",
        "processed_data_junyi",
        "processed_data_coco",
        "pam_official_static.py",
        ".runtime_tmp\PAM"
    )) {
        if ($text -notmatch [regex]::Escape($needle)) {
            throw "Expected dry-run output to contain '$needle'. Output:`n$text"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $tmpRoot) {
        Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "test_pam_official_single_seed_serial.ps1 passed"
