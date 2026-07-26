$ErrorActionPreference = "Stop"

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_STATIC_SPLIT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025"
$env:USIM_BASELINE_OUTPUT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025\main_table_fair_v1"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_SEED = "2025"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "200"

$env:CCFCREC_STATIC_EPOCHS = "80"
$env:CCFCREC_EVAL_INTERVAL = "5"
$env:CCFCREC_BATCH_SIZE = "4096"
$env:CCFCREC_EVAL_BATCH_SIZE = "4096"
$env:CCFCREC_EVAL_ITEM_MODE = "mixed"

.\py.bat ccfc_static_hin.py
.\py.bat prepare_main_table_fair_v1.py
