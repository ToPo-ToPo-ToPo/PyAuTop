
import jax.numpy as jnp
from functools import partial
from jax import jit
#=============================================================================
# nodeのクラス
#=============================================================================
class Node:
    #-------------------------------------------------------------
    # コンストラクタ
    #-------------------------------------------------------------
    def __init__(self, id, coordinate):
        self.id = id
        self.coordinate = coordinate
        self.num_dof = 0
    
    #-------------------------------------------------------------
    # 自由度番号を取得する
    # static_argnums=()で固定値の引数の番号を指定する必要がある
    #-------------------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def dof(self, idof):
        return self.num_dof * (self.id - 1) + idof