import os

import pandas as pd


def feedback_output_dir():
    explicit = os.environ.get("USIM_FB_OUTPUT_DIR", "").strip()
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    tag = os.environ.get("USIM_FB_OUTPUT_TAG", "").strip()
    if tag:
        path = os.path.join("outputs", "usim_feedback_fast3_content_delta", tag)
        os.makedirs(path, exist_ok=True)
        return path
    return "."


def feedback_output_path(filename):
    return os.path.join(feedback_output_dir(), filename)


def save_final_report_exports(
    protocol,
    metrics_keys,
    sampled_cold,
    sampled_hot,
    full_cold,
    full_hot,
    sampled_cold_count,
    sampled_hot_count,
    full_cold_count,
    full_hot_count,
    model_name="USIM-Feedback-FAST3-ContentDelta",
    full_cold_item_macro=None,
    full_hot_item_macro=None,
    full_cold_item_macro_count=0,
    full_hot_item_macro_count=0,
):
    suffix = "" if protocol == "stream" else f"_{protocol}"
    detail_path = feedback_output_path(f"final_report_usim_feedback_fast3_content_delta{suffix}.csv")
    fullrank_path = feedback_output_path(f"final_fullrank_usim_feedback_fast3_content_delta{suffix}.csv")

    detail_rows = []
    for key in metrics_keys:
        detail_rows.append(
            {
                "metric": key,
                "sampled_cold": float(sampled_cold.get(key, 0.0)) if sampled_cold_count > 0 else None,
                "sampled_hot": float(sampled_hot.get(key, 0.0)) if sampled_hot_count > 0 else None,
                "full_cold": float(full_cold.get(key, 0.0)),
                "full_hot": float(full_hot.get(key, 0.0)),
                "full_cold_item_macro": (
                    float((full_cold_item_macro or {}).get(key, 0.0))
                    if full_cold_item_macro_count > 0 else None
                ),
                "full_hot_item_macro": (
                    float((full_hot_item_macro or {}).get(key, 0.0))
                    if full_hot_item_macro_count > 0 else None
                ),
            }
        )
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    fullrank_row = {
        "model": model_name,
        "protocol": protocol,
        "full_cold_r5": float(full_cold.get("R@5", 0.0)),
        "full_cold_r10": float(full_cold.get("R@10", 0.0)),
        "full_cold_r20": float(full_cold.get("R@20", 0.0)),
        "full_cold_n5": float(full_cold.get("N@5", 0.0)),
        "full_cold_n10": float(full_cold.get("N@10", 0.0)),
        "full_cold_n20": float(full_cold.get("N@20", 0.0)),
        "full_hot_r5": float(full_hot.get("R@5", 0.0)),
        "full_hot_r10": float(full_hot.get("R@10", 0.0)),
        "full_hot_r20": float(full_hot.get("R@20", 0.0)),
        "full_hot_n5": float(full_hot.get("N@5", 0.0)),
        "full_hot_n10": float(full_hot.get("N@10", 0.0)),
        "full_hot_n20": float(full_hot.get("N@20", 0.0)),
        "full_cold_item_macro_r5": float((full_cold_item_macro or {}).get("R@5", 0.0)),
        "full_cold_item_macro_r10": float((full_cold_item_macro or {}).get("R@10", 0.0)),
        "full_cold_item_macro_r20": float((full_cold_item_macro or {}).get("R@20", 0.0)),
        "full_cold_item_macro_n5": float((full_cold_item_macro or {}).get("N@5", 0.0)),
        "full_cold_item_macro_n10": float((full_cold_item_macro or {}).get("N@10", 0.0)),
        "full_cold_item_macro_n20": float((full_cold_item_macro or {}).get("N@20", 0.0)),
        "full_hot_item_macro_r5": float((full_hot_item_macro or {}).get("R@5", 0.0)),
        "full_hot_item_macro_r10": float((full_hot_item_macro or {}).get("R@10", 0.0)),
        "full_hot_item_macro_r20": float((full_hot_item_macro or {}).get("R@20", 0.0)),
        "full_hot_item_macro_n5": float((full_hot_item_macro or {}).get("N@5", 0.0)),
        "full_hot_item_macro_n10": float((full_hot_item_macro or {}).get("N@10", 0.0)),
        "full_hot_item_macro_n20": float((full_hot_item_macro or {}).get("N@20", 0.0)),
        "sampled_cold_count": int(sampled_cold_count),
        "sampled_hot_count": int(sampled_hot_count),
        "full_cold_count": int(full_cold_count),
        "full_hot_count": int(full_hot_count),
        "full_cold_item_macro_count": int(full_cold_item_macro_count),
        "full_hot_item_macro_count": int(full_hot_item_macro_count),
        "notes": f"auto-exported from {model_name} ({protocol})",
    }
    pd.DataFrame([fullrank_row]).to_csv(fullrank_path, index=False)
    return detail_path, fullrank_path


_feedback_output_dir = feedback_output_dir
_feedback_output_path = feedback_output_path
_save_final_report_exports = save_final_report_exports

