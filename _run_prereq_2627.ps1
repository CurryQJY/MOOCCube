Set-Location 'D:\DeskTop\MOOCCube'
$ErrorActionPreference = 'Continue'
$log = 'outputs\static_prereq_v2\logs\_prereq_2627_launcher.log'
"[launcher] start {0}" -f (Get-Date -Format o) *>> $log
foreach ($s in 2026, 2027) {
    "[launcher] === prereq seed $s ===" *>> $log
    & .\run_static_prereq_v2.ps1 -Mode prereq -Seeds $s -Epochs 60 *>> $log
    "[launcher] seed $s runner exit=$LASTEXITCODE" *>> $log
}
"[launcher] done {0}" -f (Get-Date -Format o) *>> $log
