@echo off
setlocal

cd /d "%~dp0"

set "ROOT=%CD%"
set "OUTDIR=%ROOT%\outputs\content_delta_pop5\course_hparam_wide_seed2025"
set "PATHS=%OUTDIR%\course_hparam_wide_multiseed_worker_latest_paths.txt"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

start "mooccube_hparam_wide_multiseed" /min /D "%ROOT%" "%ComSpec%" /d /c ""%ROOT%\run_mooccube_paper_hparam_wide_multiseed_worker.cmd""

echo PATHS=%PATHS%
