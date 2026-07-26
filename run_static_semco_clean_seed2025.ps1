$ErrorActionPreference = "Continue"
Set-Location "D:\DeskTop\MOOCCube"
$env:PYTHONUNBUFFERED = "1"
$py = "D:\Anaconda3\envs\zw\python.exe"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "background_logs\static_semco_clean_seed2025_$ts.log"
$err = "background_logs\static_semco_clean_seed2025_$ts.err"
$out = "outputs\static_semco_clean\seed2025"
$split = "outputs\content_delta_pop5\static_item_cold_balanced\strict_item_cold_balanced_thr1_seed_2025"
New-Item -ItemType Directory -Force -Path background_logs | Out-Null
# Detached python (not nested under a short-lived parent shell).
$args = @(
  "-u", "static_semco_clean.py",
  "--split-dir", $split,
  "--output-dir", $out,
  "--seed", "2025",
  "--epochs", "60",
  "--alpha", "1.5",
  "--temp", "0.10",
  "--n-neg", "64",
  "--patience", "15"
)
$p = Start-Process -FilePath $py -ArgumentList $args `
  -WorkingDirectory "D:\DeskTop\MOOCCube" `
  -RedirectStandardOutput $log `
  -RedirectStandardError $err `
  -PassThru -WindowStyle Hidden
"PID=$($p.Id)"
"LOG=$log"
"ERR=$err"
"OUT=$out"
