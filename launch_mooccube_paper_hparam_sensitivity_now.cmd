@echo off
setlocal

cd /d "%~dp0"

set "ROOT=%CD%"
set "OUTDIR=%ROOT%\outputs\content_delta_pop5\course_hparam_sensitivity_e60_3seed"
set "PATHS=%OUTDIR%\course_hparam_sensitivity_worker_latest_paths.txt"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

start "mooccube_hparam_sensitivity" /min /D "%ROOT%" "%ComSpec%" /d /c ""%ROOT%\run_mooccube_paper_hparam_sensitivity_worker.cmd""

echo PATHS=%PATHS%
