# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import torch
import logging
from logging import getLogger
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.utils.utils import set_color


def run_recbole改版1(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    # 如果数据集是ml100k的话：self.final_config_dict['data_path'] = os.path.join(current_path, '../dataset_example/' + self.dataset)
    # dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')
    # 如果不是名为ml-100k，则是在dataset/里
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print(test_data.item_list_length[0:50])
    print(len(test_data.item_list_length))
    # exit()
    print(dataset.field2id_token['item_id'])
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # 先试试用原来的model，trainer
    config2 = Config(model='kg_model', dataset='course21_1000', config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config2['seed'], config2['reproducibility'])
    init_logger(config2)
    logger = getLogger()
    logger.info(config2)
    dataset = create_dataset(config2)
    logger.info(dataset)
    train_data1, valid_data1, test_data1 = data_preparation(config, dataset)
    test_data = test_data1.dataset.inter_feat

    # model training
    # best_valid_score, best_valid_result = trainer.fit(
    #     train_data, valid_data, saved=saved, show_progress=config['show_progress']
    # )
    test_result = trainer.evaluate(test_data, load_best_model=saved, model_file='saved/kg_model-May-04-2022_09-25-41.pth',show_progress=config['show_progress'])
    # model evaluation
    # test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    return {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }
# def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
#     r""" A fast running api, which includes the complete process of
#     training and testing a model on a specified dataset
#
#     Args:
#         model (str): model name
#         dataset (str): dataset name
#         config_file_list (list): config files used to modify experiment parameters
#         config_dict (dict): parameters dictionary used to modify experiment parameters
#         saved (bool): whether to save the model
#     """
#     # configurations initialization
#     # 如果数据集是ml100k的话：self.final_config_dict['data_path'] = os.path.join(current_path, '../dataset_example/' + self.dataset)
#     # dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')
#     # 如果不是名为ml-100k，则是在dataset/里
#     config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
#     init_seed(config['seed'], config['reproducibility'])
#
#     # logger initialization
#     init_logger(config)
#     logger = getLogger()
#     logger.info(config)
#     # dataset filtering
#     # 没啥用，不用管，返回一个dataset,包含所有信息
#     dataset = create_dataset(config)
#     # print("*"*100)
#     # print(dataset)
#     # print(type(dataset))
#     logger.info(dataset)
#
#     # dataset splitting
#     # train_data, valid_data, test_data = data_preparation(config, dataset)
#     train_data, valid_data, test_data = data_preparation(config, dataset)
#     # 三个data的dataset都是同一个东西
#     # print("*******************")
#     # print(train_data.dataset.inter_feat)
#     # print("*******************")
#     # print(valid_data.dataset.inter_feat)
#     # print("*******************")
#     # print(test_data.dataset.inter_feat)
#     # exit()
#     # print(test_data.dataset.inter_feat[test_data.target_index[0:1]])
#     # print(test_data.item_list_length)
#     # print(test_data.target_index, '******************')
#
#     # print(test_data.item_list_index)
#     # print(test_data.item_list_length)
#     # test_data.item_list_length=test_data.item_list_length[:10000]
#     # test_data.target_index = test_data.target_index[:10000]
#     # test_data.uid_list = test_data.uid_list[:10000]
#     # test_data.item_list_length = test_data.item_list_length[:10000]
#     # model loading and initialization
#     # print('*'*100)
#     # print(dataset.inter_feat)
#     # print(dataset.field2id_token['user_id'])
#     # print(type(dataset.field2id_token['user_id']))
#     # print(dataset.field2token_id)
#     # np.save('dataset_remap', dataset.field2id_token['user_id'])
#     model = get_model(config['model'])(config, train_data).to(config['device'])
#     logger.info(model)
#
#     # trainer loading and initialization
#     trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
#
#     # model training
#     # best_valid_score, best_valid_result = trainer.fit(
#     #     train_data, valid_data, saved=saved, show_progress=config['show_progress']
#     # )
#
#     # model evaluation
#     # test_result = trainer.evaluate(test_data, load_best_model=saved,model_file='saved/kg_model-May-04-2022_09-25-41.pth', show_progress=config['show_progress'])
#
#     test_result = trainer.evaluate(test_data, load_best_model=saved,
#                                    model_file='saved/reproduce_case4sr-Nov-30-2022_16-01-22.pth',
#                                    show_progress=config['show_progress'])
#     logger.info(set_color('test result', 'yellow') + f': {test_result}')
#
#     return {
#         'valid_score_bigger': config['valid_metric_bigger'],
#         'test_result': test_result
#     }
def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str): model name
        dataset (str): dataset name
        config_file_list (list): config files used to modify experiment parameters
        config_dict (dict): parameters dictionary used to modify experiment parameters
        saved (bool): whether to save the model
    """
    # configurations initialization
    # 如果数据集是ml100k的话：self.final_config_dict['data_path'] = os.path.join(current_path, '../dataset_example/' + self.dataset)
    # dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')
    # 如果不是名为ml-100k，则是在dataset/里
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    # dataset filtering
    # 没啥用，不用管，返回一个dataset,包含所有信息
    dataset = create_dataset(config)
    # print("*"*100)
    # print(dataset)
    # print(type(dataset))
    logger.info(dataset)

    # dataset splitting
    # train_data, valid_data, test_data = data_preparation(config, dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    # print('*'*100)
    # print(dataset.inter_feat)
    print(dataset.field2id_token['item_id'])

    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )



    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict): parameters dictionary used to modify experiment parameters
        config_file_list (list): config files used to modify experiment parameters
        saved (bool): whether to save the model
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
