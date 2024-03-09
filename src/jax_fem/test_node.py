
import numpy as np
import jax.numpy as jnp
from node import Node
#=============================================================================
# クラスの動作テスト
#=============================================================================
# nodeの生成
# Nodeクラスのインスタンスを作成
nodes = np.empty(3, dtype=object)  # 3つの要素を持つ配列を作成

# Nodeクラスのインスタンスを生成して配列に格納
nodes[0] = Node(1, [0.0, 0.0, 0.0])
nodes[1] = Node(2, [1.0, 1.0, 1.0])
nodes[2] = Node(3, [2.0, 2.0, 2.0])

for node in nodes:
    node.num_dof = 3

# 自由度の取得
dof1 = nodes[0].dof(0)
print(dof1)
dof1 = nodes[0].dof(2)
print(dof1)
dof1 = nodes[1].dof(0)
print(dof1)

