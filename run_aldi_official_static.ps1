$ErrorActionPreference = "Stop"

$env:USIM_DATA_DIR = "processed_data_hin_clean_pop5"
$env:USIM_STATIC_SPLIT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025"
$env:USIM_BASELINE_OUTPUT_DIR = "outputs\content_delta_pop5\static_item_cold\strict_item_cold_thr1_seed_2025\main_table_fair_v1"
$env:USIM_COLD_THRESHOLD = "1"
$env:USIM_STATIC_SEED = "2025"
$env:USIM_STATIC_TEST_HISTORY = "train_only"
$env:USIM_EVAL_N_NEG = "200"

# Point this to a Python environment that has TensorFlow installed.
# Official ALDI is TensorFlow-1 style; the wrapper patches a runtime copy to
# use tensorflow.compat.v1 when a TensorFlow-2 environment is provided.
if (-not $env:ALDI_OFFICIAL_PYTHON) {
    $env:ALDI_OFFICIAL_PYTHON = "D:\DeskTop\MOOCCube\.runtime_tmp\aldi_tf_venv\Scripts\python.exe"
}

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONUNBUFFERED = "1"
$env:ALDI_OFFICIAL_WORK_DIR = ".runtime_tmp\aldi_official_static"
$env:ALDI_OFFICIAL_TF2_COMPAT_PATCH = "1"
$env:ALDI_OFFICIAL_N_TEST_USER = "2000"
$env:ALDI_OFFICIAL_TEST_BATCH_US = "512"
$env:ALDI_OFFICIAL_TEACHER_EPOCHS = "200"
$env:ALDI_OFFICIAL_TEACHER_EVAL_INTERVAL = "20"
$env:ALDI_OFFICIAL_TEACHER_BATCH_SIZE = "4096"
$env:ALDI_OFFICIAL_STATIC_EPOCHS = "100"
$env:ALDI_OFFICIAL_EVAL_INTERVAL = "5"
$env:ALDI_OFFICIAL_BATCH_SIZE = "4096"
$env:ALDI_OFFICIAL_EVAL_BATCH_SIZE = "4096"

.\py.bat aldi_official_static_hin.py
.\py.bat prepare_main_table_fair_v1.py
