from usim_feedback_fast3_content_delta import _directory_digest


def test_v1_input_directory_digest_is_stable_and_detects_content_change(tmp_path):
    root = tmp_path / "relations"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a.json").write_text('{"edge": 1}', encoding="utf-8")
    (nested / "b.csv").write_text("u,i\n0,1\n", encoding="utf-8")

    first = _directory_digest(root)
    second = _directory_digest(root)
    (nested / "b.csv").write_text("u,i\n0,2\n", encoding="utf-8")

    assert first == second
    assert _directory_digest(root) != first
