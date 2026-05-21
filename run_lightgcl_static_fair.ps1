$ErrorActionPreference = "Stop"

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_STATIC_SPLIT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025"
$env:USIM_BASELINE_OUTPUT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025\main_table_fair_v1"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_SEED = "2025"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "200"

$env:LIGHTGCL_STATIC_EPOCHS = "80"
$env:LIGHTGCL_EVAL_INTERVAL = "5"
$env:LIGHTGCL_BATCH_SIZE = "1024"
$env:LIGHTGCL_EVAL_BATCH_SIZE = "4096"
$env:LIGHTGCL_EMB_DIM = "64"
$env:LIGHTGCL_HIDDEN_DIM = "128"
$env:LIGHTGCL_N_LAYERS = "2"
$env:LIGHTGCL_SVD_RANK = "5"
$env:LIGHTGCL_LAMBDA1 = "0.2"
$env:LIGHTGCL_LAMBDA2 = "1e-7"
$env:LIGHTGCL_TEMP = "0.2"
$env:LIGHTGCL_CONTENT_WEIGHT = "1.0"

.\py.bat lightgcl_static_hin_fair.py
.\py.bat prepare_main_table_fair_v1.py
