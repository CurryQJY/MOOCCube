@echo off
setlocal enabledelayedexpansion
cd /d D:\DeskTop\MOOCCube
set "LOGDIR=background_logs\prereq_3seed_run"
if exist "%LOGDIR%" rmdir /s /q "%LOGDIR%"
mkdir "%LOGDIR%"
echo LOGDIR = %LOGDIR%
for %%S in (2025 2026 2027) do (
    echo starting seed %%S ...
    call py.bat static_content_scorer_clean.py --data-dir processed_data_hin_clean_pop5 --split-dir outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_%%S --output-dir outputs/static_content_scorer_clean/seed%%S_prereq --seed %%S --epochs 60 --batch-size 2048 > "%LOGDIR%\seed%%S.log" 2>&1
    echo seed %%S done
)
echo ALL_DONE > "%LOGDIR%\DONE.flag"
echo === ALL THREE SEEDS DONE ===
