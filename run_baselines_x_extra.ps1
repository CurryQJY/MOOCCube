# HHCoR + LightPath on MOOCCubeX (stream first)

$env:USIM_DATA_DIR = "processed_data_hin_x"
$env:PYTHONUNBUFFERED = "1"
$PYTHON = "D:\Anaconda3\envs\zw\python.exe"
$LOG = "baselines_x_log_extra.txt"

$experiments = @(
    @{ name = "HHCoR-stream";      script = "hhcor_full_hin.py" },
    @{ name = "LightPath-stream";   script = "light_path_full_hin.py" },
    @{ name = "HHCoR-static";       script = "hhcor_static_hin.py" },
    @{ name = "LightPath-static";   script = "light_path_static_hin.py" }
)

$total = $experiments.Count
$i = 0

foreach ($exp in $experiments) {
    $i++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $header = "[$i/$total] $($exp.name) | $ts | Script: $($exp.script)"
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host $header -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Add-Content -Path $LOG -Value "`n============================================"
    Add-Content -Path $LOG -Value $header
    Add-Content -Path $LOG -Value "============================================"

    & $PYTHON $exp.script 2>&1 | Tee-Object -Append -FilePath $LOG

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $($exp.name) failed with exit code $LASTEXITCODE" -ForegroundColor Red
        Add-Content -Path $LOG -Value "[ERROR] $($exp.name) failed with exit code $LASTEXITCODE"
    } else {
        Write-Host "[DONE] $($exp.name) completed successfully" -ForegroundColor Green
        Add-Content -Path $LOG -Value "[DONE] $($exp.name) completed successfully"
    }
}

Write-Host "`nAll $total extra experiments finished!" -ForegroundColor Green
