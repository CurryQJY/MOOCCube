from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest
import scipy.sparse as sp


def test_preflight_config_disables_cbi_and_test_evaluation():
    from ckg_hot_graph_preflight import PreflightConfig

    cfg = PreflightConfig.for_seed(2025)

    assert cfg.seed == 2025
    assert cfg.use_cbi is False
    assert cfg.use_simulator is False
    assert cfg.use_ppo is False
    assert cfg.use_course_rewards is False
    assert cfg.test_evaluation is False
    assert cfg.hot_r10_floor == 0.2219
    assert cfg.hot_n10_floor == 0.1442


def test_cli_builds_config_from_explicit_training_knobs():
    from ckg_hot_graph_preflight import _config_from_args, _parser

    args = _parser().parse_args(
        [
            "--split-dir",
            "outputs/split",
            "--output-dir",
            "outputs/run",
            "--checkpoint-dir",
            "checkpoints/run",
            "--emb-dim",
            "17",
            "--mlp-hidden",
            "19",
            "--layers-gprime",
            "3",
            "--layers-full",
            "4",
            "--mask-rho",
            "0.21",
            "--lambda-e",
            "1.7",
            "--tau",
            "0.6",
            "--ranking-neg-per-user",
            "7",
            "--le-max-edges",
            "11",
            "--recon-user-chunk",
            "13",
            "--lr",
            "0.002",
            "--reg-weight",
            "0.0003",
            "--cold-threshold",
            "2",
        ]
    )

    cfg = _config_from_args(args)

    assert {
        "emb_dim": cfg.emb_dim,
        "mlp_hidden": cfg.mlp_hidden,
        "layers_gprime": cfg.layers_gprime,
        "layers_full": cfg.layers_full,
        "mask_rho": cfg.mask_rho,
        "lambda_e": cfg.lambda_e,
        "tau": cfg.tau,
        "ranking_neg_per_user": cfg.ranking_neg_per_user,
        "le_max_edges": cfg.le_max_edges,
        "recon_user_chunk": cfg.recon_user_chunk,
        "lr": cfg.lr,
        "reg_weight": cfg.reg_weight,
        "cold_threshold": cfg.cold_threshold,
    } == {
        "emb_dim": 17,
        "mlp_hidden": 19,
        "layers_gprime": 3,
        "layers_full": 4,
        "mask_rho": 0.21,
        "lambda_e": 1.7,
        "tau": 0.6,
        "ranking_neg_per_user": 7,
        "le_max_edges": 11,
        "recon_user_chunk": 13,
        "lr": 0.002,
        "reg_weight": 0.0003,
        "cold_threshold": 2,
    }


def test_masked_graph_removes_every_edge_for_selected_item():
    from ckg_hot_graph_preflight import drop_item_edges

    graph = sp.csr_matrix(([1, 1, 1], ([0, 1, 1], [2, 2, 3])), shape=(2, 4))

    masked = drop_item_edges(graph, [2])

    assert masked[:, 2].nnz == 0
    assert masked[:, 3].nnz == 1


def test_overall_uses_item_counts_not_interaction_counts():
    from ckg_hot_graph_preflight import count_weighted_overall

    assert count_weighted_overall(0.4, 2, 0.1, 8) == 0.16


def test_hot_capacity_selection_requires_both_hot_metrics():
    from ckg_hot_graph_preflight import select_best_epoch

    rows = [
        {"epoch": 1, "hot_r10": 0.23, "hot_n10": 0.13},
        {"epoch": 2, "hot_r10": 0.225, "hot_n10": 0.146},
        {"epoch": 3, "hot_r10": 0.24, "hot_n10": 0.145},
    ]

    best = select_best_epoch(rows, hot_r10_floor=0.2219, hot_n10_floor=0.1442)

    assert best["epoch"] == 3
    assert best["passes_hot_floor"] is True


def test_prepare_run_dirs_allows_launcher_manifest_but_rejects_prior_result(tmp_path):
    from ckg_hot_graph_preflight import prepare_run_dirs

    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    (output_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    prepare_run_dirs(output_dir, checkpoint_dir)

    assert checkpoint_dir.is_dir()
    (output_dir / "preflight_result.json").write_text("{}", encoding="utf-8")
    try:
        prepare_run_dirs(output_dir, checkpoint_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("a prior result must prevent accidental overwrite")


def test_preflight_rejects_a_noncanonical_shared_split():
    from ckg_hot_graph_preflight import PreflightConfig, _validate_config

    cfg = replace(PreflightConfig.for_seed(2025), split_dir="outputs/not_the_shared_split")

    with pytest.raises(ValueError, match="canonical shared split"):
        _validate_config(cfg)


def test_gate_status_distinguishes_a_completed_gate_failure():
    from ckg_hot_graph_preflight import preflight_gate_status

    assert preflight_gate_status({"passed_hot_preflight": True}) == "completed"
    assert preflight_gate_status({"passed_hot_preflight": False}) == "completed_gate_failed"


def test_train_popularity_alignment_needs_only_train_and_validation_rows():
    from ckg_hot_graph_preflight import align_train_popularity

    train = pd.DataFrame({"u_idx": [0, 1], "i_idx": [3, 3], "popularity": [99, 99]})
    validation = pd.DataFrame({"u_idx": [2, 3], "i_idx": [3, 4], "popularity": [99, 99]})

    aligned_train, aligned_validation = align_train_popularity(train, validation)

    assert aligned_train["popularity"].tolist() == [2, 2]
    assert aligned_validation["popularity"].tolist() == [2, 0]


def test_launcher_locks_validation_only_seed_2025_preflight():
    source = Path("run_ckg_hot_graph_preflight_seed2025.ps1").read_text(encoding="utf-8")

    assert "Seeds = @(2025)" in source
    assert 'OutputRoot = "outputs\\ckg_hot_graph_preflight_seed2025"' in source
    assert "TestEvaluation = $false" in source
    assert "UseCbi = $false" in source
    assert "$LASTEXITCODE" in source
    assert "validation_epochs" in source


_FORMAL_OUTPUT_ROOTS = (
    "outputs/ckg_hot_graph_preflight_seed2025",
    "checkpoints/ckg_hot_graph_preflight_seed2025",
    "background_logs/ckg_hot_graph_preflight_seed2025",
)


def _make_launcher_fixture_repo(tmp_path: Path, runner_body: str) -> Path:
    repo = tmp_path / "launcher_fixture"
    repo.mkdir()
    workspace = Path(__file__).resolve().parents[1]
    shutil.copy2(
        workspace / "run_ckg_hot_graph_preflight_seed2025.ps1",
        repo / "run_ckg_hot_graph_preflight_seed2025.ps1",
    )

    for relative_path in (
        "usim_feedback_fast3_content_delta.py",
        "fast3_delta/eval.py",
        "fast3_delta/config.py",
        "run_fast3_main_table_config.ps1",
        "paper_aaai27/main.tex",
        "ckg_hot_graph_preflight.py",
        "cgrc_paper_static_hin.py",
        "hin_data_common.py",
        "hin_eval_common.py",
        "lightgcn_static_hin.py",
        "processed_data_hin_clean_pop5/meta.json",
        "processed_data_hin_clean_pop5/content_emb.pt",
        "processed_data_hin_clean_pop5/stream_data.pkl",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_train.pkl",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_val.pkl",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_test.pkl",
        "outputs/content_delta_pop5/static_item_cold_balanced/strict_item_cold_balanced_thr1_seed_2025/static_split_assignments.csv",
    ):
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    (repo / "py.bat").write_text(runner_body, encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "preflight-test@example.invalid"),
        ("git", "config", "user.name", "Preflight Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    return repo


def _run_launcher(repo: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "run_ckg_hot_graph_preflight_seed2025.ps1"),
            "-Repo",
            str(repo),
            *extra_args,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("existing_root", _FORMAL_OUTPUT_ROOTS)
def test_launcher_rejects_each_preexisting_formal_root_before_writing_manifest(tmp_path, existing_root):
    repo = _make_launcher_fixture_repo(tmp_path, "@echo off\r\nexit /b 0\r\n")
    (repo / existing_root).mkdir(parents=True)

    completed = _run_launcher(repo)

    assert completed.returncode != 0
    assert not (repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json").exists()
    for root in _FORMAL_OUTPUT_ROOTS:
        if root != existing_root:
            assert not (repo / root).exists()


def test_launcher_dry_run_does_not_create_formal_roots(tmp_path):
    repo = _make_launcher_fixture_repo(tmp_path, "@echo off\r\nexit /b 0\r\n")

    completed = _run_launcher(repo, "-DryRun")

    assert completed.returncode == 0
    assert all(not (repo / root).exists() for root in _FORMAL_OUTPUT_ROOTS)


def test_launcher_dry_run_locks_all_fixed_python_training_knobs(tmp_path):
    repo = _make_launcher_fixture_repo(tmp_path, "@echo off\r\nexit /b 0\r\n")

    completed = _run_launcher(repo, "-DryRun")

    locked_config = json.loads(completed.stdout)
    assert locked_config["python_training_knobs"] == {
        "emb_dim": 64,
        "mlp_hidden": 64,
        "layers_gprime": 2,
        "layers_full": 2,
        "mask_rho": 0.30,
        "lambda_e": 1.0,
        "tau": 0.50,
        "ranking_neg_per_user": 32,
        "le_max_edges": 4096,
        "recon_user_chunk": 4096,
        "lr": 1e-3,
        "reg_weight": 1e-4,
        "cold_threshold": 1,
    }


def test_launcher_passes_all_locked_training_knobs_to_python(tmp_path):
    received_args = "received_python_args.txt"
    repo = _make_launcher_fixture_repo(
        tmp_path,
        f"@echo off\r\n> \"{received_args}\" echo %*\r\nexit /b 13\r\n",
    )

    completed = _run_launcher(repo)

    args = (repo / received_args).read_text(encoding="utf-8")
    assert completed.returncode != 0
    for flag, value in {
        "--emb-dim": "64",
        "--mlp-hidden": "64",
        "--layers-gprime": "2",
        "--layers-full": "2",
        "--mask-rho": "0.3",
        "--lambda-e": "1",
        "--tau": "0.5",
        "--ranking-neg-per-user": "32",
        "--le-max-edges": "4096",
        "--recon-user-chunk": "4096",
        "--lr": "0.001",
        "--reg-weight": "0.0001",
        "--cold-threshold": "1",
    }.items():
        assert f"{flag} {value}" in args


def test_failed_launcher_preserves_epochs_and_rejects_split_drift(tmp_path):
    split_val = (
        "outputs\\content_delta_pop5\\static_item_cold_balanced\\"
        "strict_item_cold_balanced_thr1_seed_2025\\static_val.pkl"
    )
    epochs = "outputs\\ckg_hot_graph_preflight_seed2025\\validation_epochs.csv"
    runner_body = (
        "@echo off\r\n"
        "if not exist \"outputs\\ckg_hot_graph_preflight_seed2025\" mkdir \"outputs\\ckg_hot_graph_preflight_seed2025\"\r\n"
        f"> \"{epochs}\" echo epoch,hot_r10,hot_n10\r\n"
        f">> \"{epochs}\" echo 1,0.10,0.05\r\n"
        f">> \"{split_val}\" echo changed-during-run\r\n"
        "exit /b 13\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed.returncode != 0
    assert manifest["status"] == "failed"
    assert manifest["validation_epochs"] == [
        {"epoch": "1", "hot_r10": "0.10", "hot_n10": "0.05"}
    ]
    assert manifest["split_sha256_after"] != manifest["split_sha256"]
    assert "Static split changed while running" in manifest["error"]


def test_launcher_hashes_consumed_data_and_rejects_data_drift(tmp_path):
    data_meta = "processed_data_hin_clean_pop5\\meta.json"
    runner_body = (
        "@echo off\r\n"
        f">> \"{data_meta}\" echo changed-during-run\r\n"
        "exit /b 13\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected_data_files = {
        "processed_data_hin_clean_pop5\\meta.json",
        "processed_data_hin_clean_pop5\\content_emb.pt",
        "processed_data_hin_clean_pop5\\stream_data.pkl",
    }
    assert completed.returncode != 0
    assert set(manifest["data_sha256"]) == expected_data_files
    assert set(manifest["data_sha256_after"]) == expected_data_files
    assert manifest["data_sha256_after"] != manifest["data_sha256"]
    assert "Consumed data changed while running" in manifest["error"]


def test_launcher_rejects_missing_consumed_input_before_invoking_python(tmp_path):
    runner_marker = "runner_was_called.txt"
    repo = _make_launcher_fixture_repo(
        tmp_path,
        f"@echo off\r\n> \"{runner_marker}\" echo invoked\r\nexit /b 0\r\n",
    )
    (repo / "processed_data_hin_clean_pop5/stream_data.pkl").unlink()

    completed = _run_launcher(repo)

    assert completed.returncode != 0
    assert not (repo / runner_marker).exists()
    assert not (repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json").exists()


def test_launcher_completed_gate_failed_exits_two_and_records_validation(tmp_path):
    epochs = "outputs\\ckg_hot_graph_preflight_seed2025\\validation_epochs.csv"
    result = "outputs\\ckg_hot_graph_preflight_seed2025\\preflight_result.json"
    runner_body = (
        "@echo off\r\n"
        f"> \"{epochs}\" echo epoch,hot_r10,hot_n10\r\n"
        f">> \"{epochs}\" echo 1,0.10,0.05\r\n"
        f"> \"{result}\" echo {{\"gate_status\":\"completed_gate_failed\",\"passed_hot_preflight\":false}}\r\n"
        "exit /b 0\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed.returncode == 2
    assert manifest["status"] == "completed_gate_failed"
    assert manifest["exit_code"] == 2
    assert manifest["gate_status"] == "completed_gate_failed"
    assert manifest["validation_epochs"] == [
        {"epoch": "1", "hot_r10": "0.10", "hot_n10": "0.05"}
    ]


def test_launcher_accepts_exit_zero_python_stderr_warning_and_logs_it(tmp_path):
    epochs = "outputs\\ckg_hot_graph_preflight_seed2025\\validation_epochs.csv"
    result = "outputs\\ckg_hot_graph_preflight_seed2025\\preflight_result.json"
    warning = "UserWarning: sparse CSR tensor support is beta"
    runner_body = (
        "@echo off\r\n"
        f"echo {warning} 1>&2\r\n"
        f"> \"{epochs}\" echo epoch,hot_r10,hot_n10\r\n"
        f">> \"{epochs}\" echo 1,0.30,0.20\r\n"
        f"> \"{result}\" echo {{\"gate_status\":\"completed\",\"passed_hot_preflight\":true}}\r\n"
        "exit /b 0\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    log_path = repo / "background_logs/ckg_hot_graph_preflight_seed2025/training.log"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed.returncode == 0
    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    assert warning in log_path.read_text(encoding="utf-16")


@pytest.mark.parametrize(
    ("result_json", "error_fragment"),
    (
        ("{}", "passed_hot_preflight"),
        (
            '{"passed_hot_preflight":"false","gate_status":"completed_gate_failed"}',
            "JSON Boolean",
        ),
        (
            '{"passed_hot_preflight":0,"gate_status":"completed_gate_failed"}',
            "JSON Boolean",
        ),
        (
            '{"passed_hot_preflight":false,"gate_status":"completed"}',
            "gate_status",
        ),
    ),
)
def test_launcher_rejects_invalid_preflight_result_contract(tmp_path, result_json, error_fragment):
    epochs = "outputs\\ckg_hot_graph_preflight_seed2025\\validation_epochs.csv"
    result = "outputs\\ckg_hot_graph_preflight_seed2025\\preflight_result.json"
    runner_body = (
        "@echo off\r\n"
        f"> \"{epochs}\" echo epoch,hot_r10,hot_n10\r\n"
        f">> \"{epochs}\" echo 1,0.10,0.05\r\n"
        f"> \"{result}\" echo {result_json}\r\n"
        "exit /b 0\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed.returncode != 0
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
    assert error_fragment in manifest["error"]


@pytest.mark.parametrize(
    ("removed_path", "after_field", "error_fragment"),
    (
        ("fast3_delta\\eval.py", "protected_files_after", "Protected files changed"),
        ("cgrc_paper_static_hin.py", "source_sha256_after", "Experiment source changed"),
        (
            "outputs\\content_delta_pop5\\static_item_cold_balanced\\"
            "strict_item_cold_balanced_thr1_seed_2025\\static_val.pkl",
            "split_sha256_after",
            "Static split changed",
        ),
        (
            "processed_data_hin_clean_pop5\\stream_data.pkl",
            "data_sha256_after",
            "Consumed data changed",
        ),
    ),
)
def test_launcher_records_deleted_integrity_inputs_and_writes_final_manifest(
    tmp_path, removed_path, after_field, error_fragment
):
    epochs = "outputs\\ckg_hot_graph_preflight_seed2025\\validation_epochs.csv"
    runner_body = (
        "@echo off\r\n"
        f"> \"{epochs}\" echo epoch,hot_r10,hot_n10\r\n"
        f">> \"{epochs}\" echo 1,0.10,0.05\r\n"
        f"del /q \"{removed_path}\"\r\n"
        "exit /b 13\r\n"
    )
    repo = _make_launcher_fixture_repo(tmp_path, runner_body)

    completed = _run_launcher(repo)

    manifest_path = repo / "outputs/ckg_hot_graph_preflight_seed2025/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed.returncode != 0
    assert manifest["status"] == "failed"
    assert manifest["validation_epochs"] == [
        {"epoch": "1", "hot_r10": "0.10", "hot_n10": "0.05"}
    ]
    assert manifest[after_field][removed_path] == "<missing>"
    assert error_fragment in manifest["error"]
