@echo off
setlocal

cd /d "%~dp0"

set "ROOT=%CD%"
set "OUTDIR=%ROOT%\outputs\content_delta_pop5\course_hparam_sim_steps_e60_3seed"
set "PATHS=%OUTDIR%\course_hparam_sim_steps_worker_latest_paths.txt"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

start "mooccube_hparam_sim_steps" /min /D "%ROOT%" "%ComSpec%" /d /c ""%ROOT%\run_mooccube_paper_hparam_sim_steps_worker.cmd""

echo PATHS=%PATHS%
