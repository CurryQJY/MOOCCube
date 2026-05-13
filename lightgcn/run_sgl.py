from recbole.quick_start import run_recbole

# 运行 SGL 模型
# if __name__ == '__main__':
#     run_recbole(
#         model='SGL',
#         dataset='MOOCCubeX',
#         config_file_list=['sgl_config.yaml']
#     )

if __name__ == '__main__':
    run_recbole(
        model='SGL',
        dataset='MOOCCubeX_Paper',
        config_file_list=['sgl_config.yaml']
    )
