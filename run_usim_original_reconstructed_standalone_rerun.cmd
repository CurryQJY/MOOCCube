@echo off
cd /d "%~dp0"

set USIM_CANDIDATE_STRATEGY=retrieve_sample
set USIM_USE_MIXED_HARD_NEG=0
set USIM_USE_EPOCH_EARLY_STOP=0
set USIM_TRAIN_FORCE_COLD=0
set USIM_RESUME=1
set USIM_CHECKPOINT=1

(
  echo Started rerun at %DATE% %TIME%
  echo Script: usim_original_reconstructed_standalone.py
  echo USIM_CANDIDATE_STRATEGY=%USIM_CANDIDATE_STRATEGY%
  echo USIM_USE_MIXED_HARD_NEG=%USIM_USE_MIXED_HARD_NEG%
  echo USIM_USE_EPOCH_EARLY_STOP=%USIM_USE_EPOCH_EARLY_STOP%
  echo USIM_TRAIN_FORCE_COLD=%USIM_TRAIN_FORCE_COLD%
  echo USIM_RESUME=%USIM_RESUME%
  echo USIM_CHECKPOINT=%USIM_CHECKPOINT%
  echo.
  echo Before Python at %DATE% %TIME%
  call .\py.bat -u usim_original_reconstructed_standalone.py
  echo After Python at %DATE% %TIME%
  echo.
  echo Finished rerun at %DATE% %TIME%
  echo ExitCode=%ERRORLEVEL%
) >> "%~dp0usim_original_reconstructed_standalone_rerun_bg.out.log" 2>> "%~dp0usim_original_reconstructed_standalone_rerun_bg.err.log"
