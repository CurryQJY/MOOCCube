@echo off
setlocal enabledelayedexpansion
cd /d D:\DeskTop\MOOCCube
set "LOGDIR=background_logs\dropout_scan_run"
if exist "%LOGDIR%" rmdir /s /q "%LOGDIR%"
mkdir "%LOGDIR%"
echo LOGDIR = %LOGDIR%
for %%D in (025 015) do (
    echo starting dropout 0.%%D ...
    call py.bat static_content_scorer_clean.py --data-dir processed_data_hin_clean_pop5 --split-dir outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025 --output-dir outputs/static_content_scorer_clean/seed2025_drop%%D --seed 2025 --epochs 60 --batch-size 2048 --dropout-prob 0.%%D > "%LOGDIR%\drop%%D.log" 2>&1
    echo dropout 0.%%D done
)
echo ALL_DONE > "%LOGDIR%\DONE.flag"
echo === DROPOUT SCAN DONE ===
