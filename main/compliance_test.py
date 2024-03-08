import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import grad
import time

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(abspath(__file__)))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from src.command.command import Command
#----------------------------------------------------------
# コンプライアンスを計算するクラス
#----------------------------------------------------------
class Compliance:
    # 初期化
    def __init__(self, id, physics, method):
        self.physics = physics
        self.method = method
    
    # コンプライアンスを計算    
    def compute(self, rho):
        self.method.run2(rho)
        U = self.method.solution_list[-1]
        
        comp = jnp.linalg.norm(U)
        
        return comp
       
#----------------------------------------------------------
# メイン
#----------------------------------------------------------
# commandクラスのオブジェクトを作成し、解析内容をテキストファイルから取得する
command = Command()
analysis_list = command.make_analysis(parent_dir + "/input/command")

# オブジェクトを作成する
physics = analysis_list[0].physics

# 解析モデルを作成する
nodes, elems, bound = physics.create_model()

# 解析クラスmethodのオブジェクトを作成する
# 線形解析の有限要素法を使用する
method = analysis_list[0].method

# コンプライアンス
compliance = Compliance(1, physics=physics, method=method)

# 感度の定義
df = grad(compliance.compute)  # 勾配関数を取得

# 感度を計算
s = np.ones(5)
print(df(s)) 