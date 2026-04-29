from recbole.quick_start import run_recbole

if __name__ == '__main__':
    run_recbole(
        model='BPR',   # 在 RecBole 中，BPR 就是最经典的 MF 实现
        dataset='MOOCCubeX_paper',
        config_file_list=['mf_config.yaml']
    )