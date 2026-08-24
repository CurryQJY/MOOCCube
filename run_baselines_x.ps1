# MOOCCubeX 基线批量实验脚本
# 5 个基线 × 2 协议 (static + stream) = 10 个实验

$env:USIM_DATA_DIR = "processed_data_hin_x"
$PYTHON = "D:\Anaconda3\envs\zw\python.exe"

$experiments = @(
    @{ name = "DropoutNet-static";  script = "drop_static_hin.py" },
    @{ name = "DropoutNet-stream";  script = "drop_full_hin.py" },
    @{ name = "GAR-static";         script = "gar_static_hin.py" },
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
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "[$i/$total] $($exp.name) | $ts" -ForegroundColor Cyan
    Write-Host "  Script: $($exp.script)" -ForegroundColor Cyan
    Write-Host "  Data:   $env:USIM_DATA_DIR" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    
    & $PYTHON $exp.script 2>&1 | Tee-Object -Append -FilePath "baselines_x_log.txt"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $($exp.name) failed with exit code $LASTEXITCODE" -ForegroundColor Red
    } else {
        Write-Host "[DONE] $($exp.name) completed successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "All $total experiments finished!" -ForegroundColor Green
