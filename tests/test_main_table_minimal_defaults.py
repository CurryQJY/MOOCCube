from fast3_delta.config import Fast3Config


def test_main_table_defaults_disable_unused_optional_components(monkeypatch):
    for key in (
        "USIM_USE_CONTENT_DELTA",
        "USIM_USE_PAAC",
        "USIM_USE_SAGE_LITE",
        "USIM_USE_SAGE_AUX_LOSS",
        "USIM_USE_CGRC_RECON",
        "USIM_USE_SG_URINIT",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = Fast3Config(2, 3, 4)

    assert cfg.use_content_delta is False
    assert cfg.use_paac is False
    assert cfg.use_sage_lite is False
    assert cfg.use_sage_aux_loss is False
    assert cfg.use_cgrc_recon is False
    assert cfg.use_sg_urinit is False
