import json
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd


COMPARISON_CSV = "comparison_stream_static_fullranking_20260314.csv"


def _metric(d, key):
    if not isinstance(d, dict):
        return 0.0
    val = d.get(key, 0.0)
    try:
        return float(val)
    except Exception:
        return 0.0


def _load_json_row(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty json list: {path}")
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported json format: {path}")


def _build_row(model_name, protocol, result, note):
    fc = result.get("full_cold", {}) or {}
    fh = result.get("full_hot", {}) or {}
    return {
        "model": model_name,
        "protocol": protocol,
        "full_cold_r5": _metric(fc, "R@5"),
        "full_cold_r10": _metric(fc, "R@10"),
        "full_cold_r20": _metric(fc, "R@20"),
        "full_cold_n5": _metric(fc, "N@5"),
        "full_cold_n10": _metric(fc, "N@10"),
        "full_cold_n20": _metric(fc, "N@20"),
        "full_hot_r5": _metric(fh, "R@5"),
        "full_hot_r10": _metric(fh, "R@10"),
        "full_hot_r20": _metric(fh, "R@20"),
        "full_hot_n5": _metric(fh, "N@5"),
        "full_hot_n10": _metric(fh, "N@10"),
        "full_hot_n20": _metric(fh, "N@20"),
        "notes": note,
    }


def _run_script(script_name, extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [sys.executable, script_name]
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main():
    skip_existing = os.environ.get("RUN_SKIP_EXISTING", "0") == "1"
    # Defaults are set for practical batch runtime while keeping training enabled.
    run_plan = [
        {
            "model": "HHCoR",
            "protocol": "static",
            "script": "hhcor_static_hin.py",
            "json": "hhcor_static_result.json",
            "env": {
                "HHCOR_STATIC_EPOCHS": os.environ.get("RUN_HHCOR_STATIC_EPOCHS", "3"),
                "HHCOR_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (HHCoR static)",
        },
        {
            "model": "HHCoR",
            "protocol": "stream",
            "script": "hhcor_full_hin.py",
            "json": "hhcor_full_result.json",
            "env": {
                "HHCOR_FULL_EPOCHS": os.environ.get("RUN_HHCOR_FULL_EPOCHS", "1"),
                "HHCOR_USE_CUMULATIVE": os.environ.get("RUN_HHCOR_FULL_CUMULATIVE", "0"),
                "HHCOR_PERIOD_TYPE": os.environ.get("RUN_PERIOD_TYPE", "Y"),
                "HHCOR_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (HHCoR stream)",
        },
        {
            "model": "LIGHT",
            "protocol": "static",
            "script": "light_path_static_hin.py",
            "json": "light_path_static_result.json",
            "env": {
                "LIGHT_STATIC_EPOCHS": os.environ.get("RUN_LIGHT_STATIC_EPOCHS", "3"),
                "LIGHT_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (LIGHT static)",
        },
        {
            "model": "LIGHT",
            "protocol": "stream",
            "script": "light_path_full_hin.py",
            "json": "light_path_full_result.json",
            "env": {
                "LIGHT_FULL_EPOCHS": os.environ.get("RUN_LIGHT_FULL_EPOCHS", "1"),
                "LIGHT_USE_CUMULATIVE": os.environ.get("RUN_LIGHT_FULL_CUMULATIVE", "0"),
                "LIGHT_PERIOD_TYPE": os.environ.get("RUN_PERIOD_TYPE", "Y"),
                "LIGHT_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (LIGHT stream)",
        },
        {
            "model": "LightGCN",
            "protocol": "static",
            "script": "lightgcn_static_hin.py",
            "json": "lightgcn_static_result.json",
            "env": {
                "LIGHTGCN_STATIC_EPOCHS": os.environ.get("RUN_LIGHTGCN_STATIC_EPOCHS", "3"),
                "LIGHTGCN_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (LightGCN static)",
        },
        {
            "model": "LightGCN",
            "protocol": "stream",
            "script": "lightgcn_full_hin.py",
            "json": "lightgcn_full_result.json",
            "env": {
                "LIGHTGCN_FULL_EPOCHS": os.environ.get("RUN_LIGHTGCN_FULL_EPOCHS", "1"),
                "LIGHTGCN_USE_CUMULATIVE": os.environ.get("RUN_LIGHTGCN_FULL_CUMULATIVE", "0"),
                "LIGHTGCN_PERIOD_TYPE": os.environ.get("RUN_PERIOD_TYPE", "Y"),
                "LIGHTGCN_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (LightGCN stream)",
        },
        {
            "model": "BERT4Rec",
            "protocol": "static",
            "script": "bert4rec_static_hin.py",
            "json": "bert4rec_static_result.json",
            "env": {
                "BERT4REC_STATIC_EPOCHS": os.environ.get("RUN_BERT4REC_STATIC_EPOCHS", "3"),
                "BERT4REC_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (BERT4Rec static)",
        },
        {
            "model": "BERT4Rec",
            "protocol": "stream",
            "script": "bert4rec_full_hin.py",
            "json": "bert4rec_full_result.json",
            "env": {
                "BERT4REC_FULL_EPOCHS": os.environ.get("RUN_BERT4REC_FULL_EPOCHS", "1"),
                "BERT4REC_USE_CUMULATIVE": os.environ.get("RUN_BERT4REC_FULL_CUMULATIVE", "0"),
                "BERT4REC_PERIOD_TYPE": os.environ.get("RUN_PERIOD_TYPE", "Y"),
                "BERT4REC_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (BERT4Rec stream)",
        },
        {
            "model": "CL4SRec",
            "protocol": "static",
            "script": "cl4srec_static_hin.py",
            "json": "cl4srec_static_result.json",
            "env": {
                "CL4SREC_STATIC_EPOCHS": os.environ.get("RUN_CL4SREC_STATIC_EPOCHS", "3"),
                "CL4SREC_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (CL4SRec static)",
        },
        {
            "model": "CL4SRec",
            "protocol": "stream",
            "script": "cl4srec_full_hin.py",
            "json": "cl4srec_full_result.json",
            "env": {
                "CL4SREC_FULL_EPOCHS": os.environ.get("RUN_CL4SREC_FULL_EPOCHS", "1"),
                "CL4SREC_USE_CUMULATIVE": os.environ.get("RUN_CL4SREC_FULL_CUMULATIVE", "0"),
                "CL4SREC_PERIOD_TYPE": os.environ.get("RUN_PERIOD_TYPE", "Y"),
                "CL4SREC_EVAL_N_NEG": os.environ.get("RUN_EVAL_N_NEG", "200"),
            },
            "note": "batch-run via run_sota_hin.py (CL4SRec stream)",
        },
    ]

    rows = []
    for job in run_plan:
        if skip_existing and os.path.exists(job["json"]):
            print(f"\n[SKIP] use existing {job['json']}")
        else:
            _run_script(job["script"], job["env"])
        result = _load_json_row(job["json"])
        row = _build_row(job["model"], job["protocol"], result, job["note"])
        rows.append(row)

    new_df = pd.DataFrame(rows)
    if os.path.exists(COMPARISON_CSV):
        old_df = pd.read_csv(COMPARISON_CSV)
    else:
        old_df = pd.DataFrame(columns=new_df.columns)

    old_key = old_df["model"].astype(str) + "||" + old_df["protocol"].astype(str) if len(old_df) > 0 else pd.Series([], dtype=str)
    new_key = new_df["model"].astype(str) + "||" + new_df["protocol"].astype(str)
    keep_old = old_df.loc[~old_key.isin(set(new_key))] if len(old_df) > 0 else old_df
    merged = pd.concat([keep_old, new_df], ignore_index=True)
    merged.to_csv(COMPARISON_CSV, index=False, float_format="%.4f")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_csv = f"comparison_stream_static_fullranking_{ts}.csv"
    merged.to_csv(backup_csv, index=False, float_format="%.4f")

    print("\n[OK] Updated comparison table:")
    print(f"  - {COMPARISON_CSV}")
    print(f"  - backup: {backup_csv}")
    print("\nNew/updated rows:")
    print(new_df.to_string(index=False))


if __name__ == "__main__":
    main()
