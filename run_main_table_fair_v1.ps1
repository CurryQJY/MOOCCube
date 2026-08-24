$ErrorActionPreference = "Stop"

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_STATIC_SPLIT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025"
$env:USIM_BASELINE_OUTPUT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025\main_table_fair_v1"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_SEED = "2025"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "200"
$env:BASELINE_BEST_METRIC = "cold"

New-Item -ItemType Directory -Force -Path $env:USIM_BASELINE_OUTPUT_DIR | Out-Null

Write-Host "== Popularity =="
.\py.bat popularity_static.py

Write-Host "== ContentProfile =="
.\py.bat content_profile_static_hin.py

Write-Host "== BPR =="
$env:BPR_STATIC_EPOCHS = "200"
$env:BPR_EVAL_INTERVAL = "5"
$env:BPR_BATCH_SIZE = "4096"
.\py.bat bpr_static_fair.py

Write-Host "== LightGCN =="
$env:LIGHTGCN_STATIC_EPOCHS = "100"
$env:LIGHTGCN_EVAL_INTERVAL = "5"
$env:LIGHTGCN_BATCH_SIZE = "4096"
.\py.bat lightgcn_static_hin_fair.py

Write-Host "== DropoutNet =="
$env:DROPOUT_STATIC_EPOCHS = "60"
.\py.bat drop_static_hin.py

Write-Host "== GAR =="
$env:GAR_STATIC_EPOCHS = "80"
$env:GAR_REC_MODE = "real_fake"
$env:GAR_EVAL_ITEM_MODE = "mixed"
.\py.bat gar_static_hin.py

Write-Host "== ALDI =="
$env:ALDI_TEACHER_EPOCHS = "200"
$env:ALDI_TEACHER_EVAL_INTERVAL = "20"
$env:ALDI_STATIC_EPOCHS = "100"
$env:ALDI_EVAL_INTERVAL = "5"
$env:ALDI_BATCH_SIZE = "4096"
.\py.bat aldi_static_hin.py

Write-Host "== Prepare summary =="
.\py.bat prepare_main_table_fair_v1.py
