@echo off
setlocal

cd /d "%~dp0"

set "ROOT=%CD%"
set "OUTDIR=%ROOT%\outputs\content_delta_pop5\course_hparam_wide_seed2025"
set "RUN_ID=%RANDOM%_%RANDOM%"
set "STDOUT=%OUTDIR%\course_hparam_wide_remaining_queue.log"
set "STDERR=%OUTDIR%\course_hparam_wide_remaining_%RUN_ID%_stderr.log"
set "MARKER=%OUTDIR%\course_hparam_wide_remaining_worker_start_marker.log"
set "PATHS=%OUTDIR%\course_hparam_wide_remaining_worker_latest_paths.txt"
set "VARIANT_LIST=sample_beta_0p00,sample_beta_0p05,sample_beta_0p15,sample_beta_0p25,sample_beta_0p40,sample_beta_0p50,reward_scale_0p00,reward_scale_0p25,reward_scale_2p00"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo [%DATE% %TIME%] worker started > "%MARKER%"
echo STDOUT=%STDOUT% > "%PATHS%"
echo STDERR=%STDERR% >> "%PATHS%"
echo VARIANT_LIST=%VARIANT_LIST% >> "%PATHS%"
echo SystemRoot=%SystemRoot% >> "%MARKER%"

set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\WindowsPowerShell\v1.0;D:\Anaconda3\envs\zw;D:\Anaconda3\envs\zw\Scripts;%ROOT%"
echo [%DATE% %TIME%] path prepared >> "%MARKER%"
echo [%DATE% %TIME%] launching remaining wide-grid seed queue >> "%MARKER%"

if defined WIDE_HPARAM_REMAINING_WORKER_PROBE (
    echo [%DATE% %TIME%] probe exit before powershell >> "%MARKER%"
    exit /b 0
)

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\run_mooccube_paper_hparam_wide_multiseed_serial.ps1" -Repo "%ROOT%" -PythonRunner "D:\Anaconda3\envs\zw\python.exe" -SeedList "2026,2027" -VariantList "%VARIANT_LIST%" -PollSeconds 300 -MinFreeGpuMiB 9000 1> "%STDOUT%" 2> "%STDERR%"
echo [%DATE% %TIME%] powershell exited with %ERRORLEVEL% >> "%MARKER%"
