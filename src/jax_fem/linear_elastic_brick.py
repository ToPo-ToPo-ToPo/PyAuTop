
import jax.numpy as jnp
from functools import partial
from jax import jit
#=============================================================================
# 3次元ソリッド要素に対する等方性弾生体モデルの構成則を計算するためのクラス
#=============================================================================
class LinearElasticBrick:
    # コンストラクタ
    # young       : ヤング率
    # poisson     : ポアソン比
    # density     : 密度
    def __init__(self, young, poisson, density):
        # インスタンス変数を定義する
        self.young = young                           # ヤング率
        self.poisson = poisson                       # ポアソン比
        self.density = density                       # 密度

        # 関連する変数を初期化する
        self.strain = jnp.zeros(6)                   # 要素内の弾性ひずみ
        self.stress = jnp.zeros(6)                   # 要素内の応力
        self.mises = 0.0                             # 要素内のミーゼス応力

    #---------------------------------------------------------------------
    # 応力の計算
    # Ue : 要素節点の変位ベクトル(jnp.array型)
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1, 2))
    def compute_stress(self, B, Ue, x):
        # 全ひずみを求める
        strain = B @ Ue 
        
        # 弾性剛性マトリクスを作成する
        C = self.make_C(x)
        
        # 応力を求める
        stress = C @ strain
        
        return stress
    
    #---------------------------------------------------------------------
    # 接線剛性の計算
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0, 1, 2))
    def compute_tangent_matrix(self, B, Ue, x):
        return self.make_C(x)
    
    #---------------------------------------------------------------------
    # ニュートンラプソン法収束後の内部変数の更新
    #---------------------------------------------------------------------
    def update(self):
        pass
    
    #---------------------------------------------------------------------
    # 弾性状態のDマトリクスを作成する
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0))
    def make_C(self, x):
        #
        young = (1.0e-03+x**3.0) * self.young
        #
        poisson = self.poisson
        #
        tmp = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        C0 = jnp.array([
            [1.0 - poisson, poisson, poisson, 0.0, 0.0, 0.0],
            [poisson, 1.0 - poisson, poisson, 0.0, 0.0, 0.0],
            [poisson, poisson, 1.0 - poisson, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)]
        ])
        C = tmp * C0
        return C

    #---------------------------------------------------------------------
    # ミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    #---------------------------------------------------------------------
    @partial(jit, static_argnums=(0))
    def mises_stress(self, stress):
        tmp1 = jnp.square(stress[0] - stress[1]) + jnp.square(stress[1] - stress[2]) + jnp.square(stress[2] - stress[0])
        tmp2 = 6.0 * (jnp.square(stress[3]) + jnp.square(stress[4]) + jnp.square(stress[5]))
        mises = jnp.sqrt(0.5 * (tmp1 + tmp2))
        return mises
