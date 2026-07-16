import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import paper_aaai27.scripts.analyze_p1_motivation_mechanisms as mechanisms
from paper_aaai27.scripts.analyze_p1_motivation_mechanisms import (
    MECHANISM_COMPARISONS,
    _ablation_export_paths,
    analyze_ablation_course_rows,
    build_mechanism_claim_audit,
    combine_full_and_ablation_course_rows,
    load_full_ckgrl_course_rows,
    run_analysis,
)


LIST_METRICS = (
    "prerequisite_gap",
    "concept_continuity",
    "difficulty_gap",
    "structural_redundancy",
    "cold_proportion",
    "cold_prerequisite_gap",
    "cold_concept_continuity",
    "cold_difficulty_gap",
    "cold_structural_redundancy",
)


def _course_row(model, seed=2025, target_item_id=7, cutoff=10, value=0.1):
    return {
        "model": model,
        "seed": seed,
        "target_item_id": target_item_id,
        "cutoff": cutoff,
        "list_count": 1,
        "cold_list_count": 1,
        **{metric: value for metric in LIST_METRICS},
    }


def _write_full_analysis(full_dir: Path) -> pd.DataFrame:
    full_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            _course_row(model, cutoff=cutoff, value=value)
            for model, value in (
                ("ckg_rl", 0.1),
                ("pcgnn", 0.2),
                ("cgrc", 0.3),
            )
            for cutoff in (10, 20)
        ]
    )
    course_path = full_dir / "course_macro.csv"
    rows.to_csv(course_path, index=False)
    (full_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "seeds": [2025],
                "cutoffs": [10, 20],
                "top_k": 20,
                "outputs": {"course_macro": str(course_path.resolve())},
            }
        ),
        encoding="utf-8",
    )
    return rows


def test_ablation_export_paths_use_labeled_ckgrl_directories(tmp_path):
    paths = _ablation_export_paths(tmp_path, 2026)

    assert set(paths) == {
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    }
    for model, path in paths.items():
        assert path == (
            tmp_path
            / "outputs"
            / "p1_motivation_topk"
            / model
            / "strict_item_cold_balanced_thr1_seed_2026"
            / "top20_cold_test.jsonl"
        )


def test_mechanism_comparisons_treat_the_full_model_as_treatment():
    assert MECHANISM_COMPARISONS == (
        ("ckg_rl", "ckg_rl_wo_course_reward", "course_reward"),
        ("ckg_rl", "ckg_rl_wo_simulator", "simulator"),
    )


def test_course_row_combination_keeps_one_row_per_model_seed_course_cutoff():
    full = pd.DataFrame(
        [
            {
                "model": "ckg_rl",
                "seed": 2025,
                "target_item_id": 7,
                "cutoff": 10,
                "prerequisite_gap": 0.1,
            },
            {
                "model": "cgrc",
                "seed": 2025,
                "target_item_id": 7,
                "cutoff": 10,
                "prerequisite_gap": 0.2,
            },
        ]
    )
    ablations = pd.DataFrame(
        [
            {
                "model": model,
                "seed": 2025,
                "target_item_id": 7,
                "cutoff": 10,
                "prerequisite_gap": value,
            }
            for model, value in (
                ("ckg_rl_wo_course_reward", 0.3),
                ("ckg_rl_wo_simulator", 0.4),
            )
        ]
    )

    combined = combine_full_and_ablation_course_rows(full, ablations)

    assert combined["model"].tolist() == [
        "ckg_rl",
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    ]

    duplicated = pd.concat([ablations, ablations.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate mechanism course rows"):
        combine_full_and_ablation_course_rows(full, duplicated)


def test_course_row_combination_rejects_missing_ablation_pair_coverage():
    full = pd.DataFrame([_course_row("ckg_rl")])
    incomplete = pd.DataFrame([_course_row("ckg_rl_wo_course_reward")])

    with pytest.raises(ValueError, match="mechanism course coverage mismatch"):
        combine_full_and_ablation_course_rows(full, incomplete)


def test_ablation_analysis_validates_only_the_two_ablation_exports(monkeypatch, tmp_path):
    validation_calls = []
    native_calls = []

    monkeypatch.setattr(
        mechanisms,
        "_seed_inputs",
        lambda split_root, seed, n_items: (
            [(3, 7)],
            {3: np.array([0, 1], dtype=np.int64)},
            np.array([1, 1, 0], dtype=np.int64),
        ),
    )

    def fake_validate(root, *, model, seed, export_path, expected_count):
        validation_calls.append((model, seed, export_path, expected_count))
        return {"model": model, "seed": seed}

    monkeypatch.setattr(mechanisms, "validate_export_provenance", fake_validate)

    def fake_analyze(**kwargs):
        assert set(kwargs["model_paths"]) == {
            "ckg_rl_wo_course_reward",
            "ckg_rl_wo_simulator",
        }
        for model in kwargs["model_paths"]:
            for cutoff in (10, 20):
                kwargs["course_accumulator"].update(
                    _course_row(model, cutoff=cutoff, value=0.2)
                )
        return {
            model: {
                "record_count": 1,
                "target_course_count": 1,
                "R@10": 0.2,
                "N@10": 0.1,
                "course_macro_R@10": 0.2,
                "course_macro_N@10": 0.1,
                "seen_item_leak_count": 0,
                "invalid_topk_count": 0,
            }
            for model in kwargs["model_paths"]
        }

    monkeypatch.setattr(mechanisms, "analyze_seed_export_pair", fake_analyze)

    def fake_native(root, seed, model):
        native_calls.append((seed, model))
        return {
            "count_full_cold": 1,
            "R@10": 0.2,
            "N@10": 0.1,
            "course_macro_R@10": 0.2,
            "course_macro_N@10": 0.1,
        }

    monkeypatch.setattr(mechanisms, "_native_export_metrics", fake_native)

    course_rows, audit_rows, export_paths = analyze_ablation_course_rows(
        root=tmp_path,
        seeds=(2025,),
        artifacts=SimpleNamespace(structural_complexity=np.zeros(3)),
    )

    expected_models = {
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    }
    assert set(course_rows["model"]) == expected_models
    assert set(audit_rows["model"]) == expected_models
    assert {call[0] for call in validation_calls} == expected_models
    assert {model for _, model in native_calls} == expected_models
    assert len(export_paths) == 2


def test_full_course_loader_uses_only_ckgrl_from_the_bound_analysis(tmp_path):
    full_dir = tmp_path / "full_analysis"
    _write_full_analysis(full_dir)

    full_rows = load_full_ckgrl_course_rows(full_dir, seeds=(2025,))

    assert full_rows["model"].unique().tolist() == ["ckg_rl"]
    assert set(full_rows["cutoff"]) == {10, 20}

    manifest_path = full_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seeds"] = [2026]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="full analysis seed mismatch"):
        load_full_ckgrl_course_rows(full_dir, seeds=(2025,))


def test_run_analysis_records_reused_full_input_and_only_recomputed_ablations(
    monkeypatch,
    tmp_path,
):
    full_dir = tmp_path / "full_analysis"
    _write_full_analysis(full_dir)
    export_paths = []
    for model in ("ckg_rl_wo_course_reward", "ckg_rl_wo_simulator"):
        path = tmp_path / f"{model}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        export_paths.append(path)

    monkeypatch.setattr(
        mechanisms,
        "build_real_risk_artifacts",
        lambda root: (
            SimpleNamespace(structural_complexity=np.zeros(3)),
            {"n_items": 3},
        ),
    )

    ablation_rows = pd.DataFrame(
        [
            _course_row(model, cutoff=cutoff, value=value)
            for model, value in (
                ("ckg_rl_wo_course_reward", 0.2),
                ("ckg_rl_wo_simulator", 0.3),
            )
            for cutoff in (10, 20)
        ]
    )
    audit_rows = pd.DataFrame(
        [
            {"model": model, "seed": 2025, "record_count": 1}
            for model in ("ckg_rl_wo_course_reward", "ckg_rl_wo_simulator")
        ]
    )
    monkeypatch.setattr(
        mechanisms,
        "analyze_ablation_course_rows",
        lambda **kwargs: (ablation_rows, audit_rows, export_paths),
    )

    output_dir = tmp_path / "mechanism_analysis"
    manifest = run_analysis(
        root=tmp_path,
        full_analysis_dir=full_dir,
        output_dir=output_dir,
        seeds=(2025,),
        n_bootstrap=20,
        n_permutations=30,
        random_seed=2027,
    )

    assert manifest["full_model_reused"] is True
    assert manifest["recomputed_models"] == [
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    ]
    assert set(manifest["full_analysis_input_sha256"]) == {
        str((full_dir / "analysis_manifest.json").resolve()),
        str((full_dir / "course_macro.csv").resolve()),
    }
    assert manifest["row_counts"]["course_macro"] == 6
    assert manifest["row_counts"]["paired_statistics"] == 36

    combined = pd.read_csv(output_dir / "course_macro.csv")
    assert set(combined["model"]) == {
        "ckg_rl",
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    }
    paired = pd.read_csv(output_dir / "paired_statistics.csv")
    assert set(paired["treatment"]) == {"ckg_rl"}
    assert set(paired["baseline"]) == {
        "ckg_rl_wo_course_reward",
        "ckg_rl_wo_simulator",
    }


def test_claim_audit_preserves_support_inconclusive_and_adverse_results():
    paired = pd.DataFrame(
        [
            {
                "comparison_role": "course_reward",
                "cutoff": 10,
                "metric": metric,
                "direction": direction,
                "mean_difference": difference,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "permutation_p_value": p_value,
                "interpretation": interpretation,
            }
            for metric, direction, difference, low, high, p_value, interpretation in (
                (
                    "cold_prerequisite_gap",
                    "lower",
                    -0.013,
                    -0.020,
                    -0.006,
                    0.001,
                    "supports",
                ),
                (
                    "prerequisite_gap",
                    "lower",
                    -0.004,
                    -0.008,
                    0.001,
                    0.16,
                    "inconclusive",
                ),
                (
                    "concept_continuity",
                    "higher",
                    -0.002,
                    -0.003,
                    -0.001,
                    0.0002,
                    "falsifies",
                ),
                (
                    "cold_proportion",
                    "descriptive",
                    0.027,
                    0.019,
                    0.035,
                    0.00001,
                    "descriptive",
                ),
            )
        ]
    )

    audit = build_mechanism_claim_audit(paired, cutoff=10)

    assert audit["claim_status"].tolist() == [
        "supports_mechanism",
        "inconclusive",
        "adverse_tradeoff",
        "descriptive_exposure",
    ]
    assert audit.iloc[0]["favorable_effect"] == pytest.approx(0.013)
    assert audit.iloc[2]["favorable_effect"] == pytest.approx(-0.002)
