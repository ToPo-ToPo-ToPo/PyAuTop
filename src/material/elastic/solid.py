#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np
import numpy.linalg as LA
from src.material.dmatrix import Dmatrix
import jax
from jax import jit
import jax.numpy as jnp
#=============================================================================
# 3次元ソリッド要素に対する等方性弾生体モデルの構成則を計算するためのクラス
#=============================================================================
class ElasticSolid:
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
        self.vecEStrain = np.zeros(6)                # 要素内の弾性ひずみ
        self.vecStress = np.zeros(6)                 # 要素内の応力
        self.mises = 0.0                             # 要素内のミーゼス応力

        # Dマトリックスを初期化する
        #self.matD = Dmatrix(young, poisson).make_De_matrix()

    #---------------------------------------------------------------------
    # 応力を更新する
    # solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    @jit
    def compute_stress_and_tangent_matrix(self, matB, solution):
        
        # 全ひずみを求める
        vecStrain = matB @ solution 
        
        #
        matD = self.make_De_matrix()

        # 応力を求める
        self.vecStress = matD @ vecStrain
        
        # mises応力を求める
        self.mises = self.mises_stress(self.vecStress)
    
    #---------------------------------------------------------------------
    # ニュートンラプソン法収束後の内部変数の更新
    #---------------------------------------------------------------------
    @jit
    def update(self):
        pass
    
    #---------------------------------------------------------------------
    # 弾性状態のDマトリクスを作成する
    #---------------------------------------------------------------------
    def make_De_matrix(self):
        young = self.young
        poisson = self.poisson
        tmp = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        matD = jnp.array([[1.0 - poisson, poisson, poisson, 0.0, 0.0, 0.0],
                         [poisson, 1.0 - poisson, poisson, 0.0, 0.0, 0.0],
                         [poisson, poisson, 1.0 - poisson, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)]])
        matD = tmp * matD

        return matD

    #---------------------------------------------------------------------
    # ミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    #---------------------------------------------------------------------
    @jit
    def mises_stress(self, vecStress):

        tmp1 = jnp.square(vecStress[0] - vecStress[1]) + jnp.square(vecStress[1] - vecStress[2]) + jnp.square(vecStress[2] - vecStress[0])
        tmp2 = 6.0 * (np.jsquare(vecStress[3]) + jnp.square(vecStress[4]) + jnp.square(vecStress[5]))
        mises = jnp.sqrt(0.5 * (tmp1 + tmp2))

        return mises
