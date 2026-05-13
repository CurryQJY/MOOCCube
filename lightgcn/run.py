from recbole.quick_start import run_recbole

run_recbole(
    model='LightGCN',
    dataset='MOOCCubeX_Paper',
    config_file_list=['lightgcn_config.yaml']
)
