@echo off
setlocal enabledelayedexpansion
cd /d D:\DeskTop\MOOCCube
set PYTHONUNBUFFERED=1

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "LOGDIR=background_logs\prereq_w2_3seed_%TS%"
mkdir "%LOGDIR%" 2>nul
echo LOGDIR = %LOGDIR%
echo LOGDIR = %LOGDIR%> "%LOGDIR%\driver.log"

for %%S in (2025 2026 2027) do (
    echo === START seed %%S prereq-weight=2.0 %DATE% %TIME% ===
    echo === START seed %%S prereq-weight=2.0 %DATE% %TIME% ===>> "%LOGDIR%\driver.log"
    call py.bat -u graph_gated_scorer_clean.py ^
        --data-dir processed_data_hin_clean_pop5 ^
        --split-dir outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_%%S ^
        --output-dir outputs/graph_gated_scorer_clean/seed%%S_w2 ^
        --seed %%S ^
        --epochs 60 ^
        --batch-size 2048 ^
        --n-layers 2 ^
        --prereq-weight 2.0 ^
        > "%LOGDIR%\seed%%S.log" 2>&1
    echo seed %%S EXIT=!ERRORLEVEL!
    echo seed %%S EXIT=!ERRORLEVEL!>> "%LOGDIR%\driver.log"
)

echo ALL_DONE > "%LOGDIR%\DONE.flag"
echo === ALL THREE SEEDS DONE ===
echo DONE.flag written to %LOGDIR%\DONE.flag
endlocal
