# SEMCo
This repository contains code and resources for the SIGIR 2026 short paper "[Sparse Contrastive Learning for Content-Based Cold Item Recommendation](https://arxiv.org/abs/2604.12990)". The [model implementations](https://github.com/gmeehan96/SEMCo/tree/main/models) of our SEMCo variants are based on [ColdRec](https://github.com/YuanchenBei/ColdRec). We include the final hyperparameter values for each dataset in the model training scripts. To run a particular configuration, use a command of the form
```
python main_SEMCo.py --dataset [dataset_name] --fn sparsemax
```
We also include the [script](https://github.com/gmeehan96/SEMCo/blob/main/main_SEMCo_online_tpe.py) for TPE hyperparameter search on the SEMCo-Offline model.

### Hyperparameter ranges
The table below contains the hyperparameter search ranges for the key hyperparameters of each SEMCo variant.
| Hyperparameter          | Softmax                                       | Entmax                                 | Sparsemax                              | Variants Used   |
|-------------------------|-----------------------------------------------|----------------------------------------|----------------------------------------|-----------------|
| $\tau$                  | [0.1, 0.125, 0.15, 0.2,  0.25, 0.3, 0.4, 0.5] | [1.5, 2, 2.5,..., 5]                   | [6, 8, 10, ... , 20]                   | SEMCo, Online   |
| L2 Regularization       | [0.0, 0.00001, 0.0001,  0.0005, 0.001]        | [0.0, 0.00001, 0.0001,  0.0005, 0.001] | [0.0, 0.00001, 0.0001,  0.0005, 0.001] | SEMCo, Online   |
| $\omega$                | [0.1, 0.125, 0.15, 0.2,  0.25, 0.3]           | [1.5, 2, 2.5,..., 4]                   | [6, 8, 10, ... , 16]                   | Offline, Online |
| Positive examples       | [5, 10, 15]                                   | [5, 10, 15]                            | [5, 10, 15]                            | Offline, Online |
| $\lambda$               | [0.1, 0.5, 1.0, 2.0, 5.0]                     | [0.1, 0.5, 1.0, 2.0, 5.0]              | [0.1, 0.5, 1.0, 2.0, 5.0]              | Offline, Online |
| EMA Momentum            | [0.0, 0.5, 0.9, 0.99]                         | [0.0, 0.5, 0.9, 0.99]                  | [0.0, 0.5, 0.9, 0.99]                  | Online          |
| Student warmup epochs   | [0, 1, 3, 5]                                  | [0, 1, 3, 5]                           | [0, 1, 3, 5]                           | Online          |
| Student training epochs | [15, 20, 25]                                  | [15, 20, 25]                           | [15, 20, 25]                           | Online          |
