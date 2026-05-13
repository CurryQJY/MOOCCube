# MOOCCubeX 基线 - 剩余实验（已完成: DropoutNet static/stream, GAR-static）
# Static epochs 减到 20，加速训练

$env:USIM_DATA_DIR = "processed_data_hin_x"
$env:PYTHONUNBUFFERED = "1"
$PYTHON = "D:\Anaconda3\envs\zw\python.exe"
$LOG = "baselines_x_log.txt"

# Static epochs 设为 20
$env:LIGHTGCL_STATIC_EPOCHS = "20"
$env:SASREC_STATIC_EPOCHS = "20"
$env:LIGHTGCN_STATIC_EPOCHS = "8"  # LightGCN 原本就是 8，保持不变

$experiments = @(
    @{ name = "GAR-stream";         script = "gar_full_hin.py" },
    @{ name = "LightGCL-static";    script = "lightgcl_static_hin.py" },
    @{ name = "LightGCL-stream";    script = "lightgcl_full_hin.py" },
    @{ name = "LightGCN-static";    script = "lightgcn_static_hin.py" },
    @{ name = "LightGCN-stream";    script = "lightgcn_full_hin.py" },
    @{ name = "SASRec-static";      script = "sasrec_static_hin.py" },
    @{ name = "SASRec-stream";      script = "sasrec_full_hin.py" }
)

$total = $experiments.Count
$i = 0

foreach ($exp in $experiments) {
    $i++
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $header = "[$i/$total] $($exp.name) | $ts | Script: $($exp.script)"
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host $header -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    
    # 写 header 到日志
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

Write-Host ""
Write-Host "All $total remaining experiments finished!" -ForegroundColor Green
Add-Content -Path $LOG -Value "`nAll $total remaining experiments finished!"
