#!/usr/bin/env bash
# Waits for the 3-seed graph scorer (DONE.flag), then launches the CGRC formal
# timing queue as a DETACHED process and verifies it actually started.
set -u
cd /d/DeskTop/MOOCCube

LOG=background_logs/_cgrc_timing_autostart.log
DONE_FLAG=background_logs/graph_3seed_011941/DONE.flag
STAMP=$(date +%Y%m%d_%H%M%S)
TIMING_LOG="background_logs/cgrc_formal_timing_autostart_${STAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

launch_queue() {
  local out="$1"
  # Start-Process fully detaches the queue from this daemon so it survives our exit.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
    "Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-Command',\"Set-Location 'D:\DeskTop\MOOCCube'; .\run_cgrc_controlled_timing.ps1 -TimingOnly *> '${out}'\" -WindowStyle Hidden" >/dev/null 2>&1
}

verify_started() {
  local out="$1"
  local i
  for i in $(seq 1 30); do
    sleep 10
    if grep -q "START dataset=Junyi timing_seed=9101" "$out" 2>/dev/null; then
      local running
      running=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*cgrc_paper_static_hin*' } | Measure-Object).Count" 2>/dev/null | tr -d '\r ')
      if [ "${running:-0}" != "0" ]; then
        log "VERIFIED: START line present AND cgrc_paper_static_hin python running (count=$running)"
        return 0
      fi
      log "startup check $i/30: START line seen but python not up yet"
    else
      log "startup check $i/30: no START line yet"
    fi
  done
  return 1
}

log "===== autostart daemon begin (stamp $STAMP) ====="
log "waiting for $DONE_FLAG"

WAITED=0; MAX_WAIT=$((3*3600))
while [ ! -f "$DONE_FLAG" ]; do
  sleep 30; WAITED=$((WAITED+30))
  [ $((WAITED % 300)) -eq 0 ] && log "still waiting (${WAITED}s)"
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    log "ERROR: DONE.flag absent after ${MAX_WAIT}s; aborting"; exit 1
  fi
done
log "DONE.flag detected"

# Confirm the scorer python is actually gone (GPU freed).
for i in $(seq 1 24); do
  running=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*graph_content_scorer_clean*' } | Measure-Object).Count" 2>/dev/null | tr -d '\r ')
  [ "${running:-0}" = "0" ] && { log "scorer confirmed stopped"; break; }
  log "scorer still running (attempt $i); waiting"; sleep 15
done
sleep 10

log "launching timing queue -> $TIMING_LOG"
launch_queue "$TIMING_LOG"
if verify_started "$TIMING_LOG"; then
  log "SUCCESS: CGRC timing experiment running. Daemon exiting."
  exit 0
fi

log "WARN: first launch not confirmed in 5min; retrying once"
launch_queue "${TIMING_LOG}.retry"
if verify_started "${TIMING_LOG}.retry"; then
  log "SUCCESS on retry: CGRC timing experiment running. Daemon exiting."
  exit 0
fi

log "FATAL: could not confirm timing queue startup after retry. Manual help needed."
exit 1
