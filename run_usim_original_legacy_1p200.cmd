@echo off
cd /d "%~dp0"

set USIM_OUTPUT_TAG=original_legacy_1p200
set USIM_CANDIDATE_STRATEGY=random
set USIM_TRAIN_FORCE_COLD=0
set USIM_USE_MIXED_HARD_NEG=0
set USIM_USE_EPOCH_EARLY_STOP=0
set USIM_EVAL_N_NEG=200
set USIM_LEGACY_EVAL_POS_FROM_BANK=1
set USIM_RESUME=1
set USIM_CHECKPOINT=1
set USIM_STREAM_CKPT=usim_original_legacy_1p200.stream_ckpt.pt

(
  echo Started legacy 1+200 rerun at %DATE% %TIME%
  echo Script: usim_original_reconstructed_standalone.py
  echo USIM_OUTPUT_TAG=%USIM_OUTPUT_TAG%
  echo USIM_CANDIDATE_STRATEGY=%USIM_CANDIDATE_STRATEGY%
  echo USIM_TRAIN_FORCE_COLD=%USIM_TRAIN_FORCE_COLD%
  echo USIM_USE_MIXED_HARD_NEG=%USIM_USE_MIXED_HARD_NEG%
  echo USIM_USE_EPOCH_EARLY_STOP=%USIM_USE_EPOCH_EARLY_STOP%
  echo USIM_EVAL_N_NEG=%USIM_EVAL_N_NEG%
  echo USIM_LEGACY_EVAL_POS_FROM_BANK=%USIM_LEGACY_EVAL_POS_FROM_BANK%
  echo USIM_RESUME=%USIM_RESUME%
  echo USIM_CHECKPOINT=%USIM_CHECKPOINT%
  echo.
  call .\py.bat -u usim_original_reconstructed_standalone.py
  echo.
  echo Finished legacy 1+200 rerun at %DATE% %TIME%
  echo ExitCode=%ERRORLEVEL%
) >> "%~dp0usim_original_legacy_1p200.out.log" 2>> "%~dp0usim_original_legacy_1p200.err.log"
