from recbole.quick_start import run_recbole

# if __name__ == '__main__':
#     run_recbole(
#         model='SASRec',
#         dataset='MOOCCubeX',
#         config_file_list=['sasrec_config.yaml']
#     )

if __name__ == '__main__':
    run_recbole(
        model='SASRec',
        dataset='MOOCCubeX_Paper',
        config_file_list=['sasrec_config.yaml']
    )