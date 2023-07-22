
from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import numpy as np
from src.model.element.element_interface import ElementInterface 
#=============================================================================
# 要素クラスの基本クラスを定義する
# 共有部分を定義する
#=============================================================================
class ElementBase(ElementInterface):

    #---------------------------------------------------------------------
    # 要素内の自由度番号のリストを作成する
    #---------------------------------------------------------------------
    def make_dof_list(self, nodes, num_node, num_dof):
        
        dof_list = np.zeros(num_node * num_dof, dtype=np.int64)
        
        for i in range(num_node):
            for j in range(num_dof):
                dof_list[i * num_dof + j] = nodes[i].dof(j)
        
        return dof_list