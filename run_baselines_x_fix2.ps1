# 补跑 — Stream 优先
# 已完成: LightGCL(both), SASRec(both), GAR-stream, DropoutNet-static
# 剩余: DropoutNet-stream, LightGCN-stream, GAR-static, LightGCN-static

$env:USIM_DATA_DIR = "processed_data_hin_x"
$env:PYTHONUNBUFFERED = "1"
$PYTHON = "D:\Anaconda3\envs\zw\python.exe"
$LOG = "baselines_x_log_fix2.txt"

$env:LIGHTGCN_STATIC_EPOCHS = "8"

$experiments = @(
    @{ name = "DropoutNet-stream";  script = "drop_full_hin.py" },
    @{ name = "LightGCN-stream";    script = "lightgcn_full_hin.py" },
    @{ name = "GAR-static";         script = "gar_static_hin.py" },
    @{ name = "LightGCN-static";    script = "lightgcn_static_hin.py" }
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

Write-Host "`nAll $total fix2 experiments finished!" -ForegroundColor Green
