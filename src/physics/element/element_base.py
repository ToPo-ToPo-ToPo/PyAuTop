

import numpy as np
from jax import jit
import jax.numpy as jnp
from src.physics.element.element_interface import ElementInterface 
#=============================================================================
# 要素クラスの基本クラスを定義する
# 共有部分を定義する
#=============================================================================
class ElementBase(ElementInterface):

    #---------------------------------------------------------------------
    # 要素内の自由度番号のリストを作成する
    #---------------------------------------------------------------------
    def make_dof_list(self, nodes, num_node, num_dof):
        
        dof_list = jnp.zeros(num_node * num_dof, dtype=jnp.int32)
        
        for i in range(num_node):
            for j in range(num_dof):
                # jaxを使用する際の配列の更新方法
                #dof_list[i * num_dof + j] = nodes[i].dof(j)
                dof_list = dof_list.at[i * num_dof + j].set(nodes[i].dof(j))
        
        return dof_list