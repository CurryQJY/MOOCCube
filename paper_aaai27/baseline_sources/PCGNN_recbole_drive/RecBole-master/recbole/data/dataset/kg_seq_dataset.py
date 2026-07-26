# @Time   : 2020/9/23
# @Author : Xingyu Pan
# @Email  : panxingyu@ruc.edu.cn

"""
recbole.data.kg_seq_dataset
#############################
"""

from recbole.data.dataset import SequentialDataset, KnowledgeBasedDataset


class Kg_Seq_Dataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~recbole.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """

    def __init__(self, config):
        # 多继承时，相对于使用类名.__init__方法，要把每个父类全部写一遍
        # 而super只用一句话，执行了全部父类的方法，这也是为何多继承需要全部传参的一个原因
        super().__init__(config)
