# https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

# kato

import numpy as np
import numpy.linalg as LA
from src.material.dmatrix import Dmatrix

# =============================================================================
# 2次元ソリッド要素に対する等方性弾生体モデルの構成則を計算するためのクラス
# =============================================================================
class ElasticSolidPlaneStrain:
    # コンストラクタ
    # young       : ヤング率
    # poisson     : ポアソン比
    # density     : 密度
    def __init__(self, young, poisson, density):
        # インスタンス変数を定義する
        self.young = young  # ヤング率
        self.poisson = poisson  # ポアソン比
        self.density = density  # 密度

        # 関連する変数を初期化する
        self.vecEStrain = np.zeros(6)  # 要素内の弾性ひずみ
        self.vecStress = np.zeros(6)  # 要素内の応力
        self.mises = 0.0  # 要素内のミーゼス応力

        # Dマトリックスを初期化する
        self.matD = Dmatrix(young, poisson).make_De_matrix()
        
    #---------------------------------------------------------------------
    # 応力を更新する
    # solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def compute_stress_and_tangent_matrix(self, matB, solution):
        
        # 全ひずみを求める
        vecStrain = matB @ solution 
        
        # ひずみのz軸方向の成分を0にする
        vecStrain[2] = 0.0  # z方向の垂直ひずみ
        vecStrain[4] = 0.0  # yz方向のせん断ひずみ
        vecStrain[5] = 0.0  # xz方向のせん断ひずみ
        
        # 応力を求める
        self.vecStress = self.matD @ vecStrain
        
        # z方向の応力を求める
        tmp = (self.poisson * self.young) / ((1 + self.poisson) * (1 - 2 * self.poisson))
        self.vecStress[2] = tmp * (vecStrain[0] + vecStrain[1])
        
        # mises応力を求める
        self.mises = self.mises_stress(self.vecStress)
    
    #---------------------------------------------------------------------
    # ニュートンラプソン法収束後の内部変数の更新
    #---------------------------------------------------------------------
    def update(self):
        pass

    #---------------------------------------------------------------------
    # ミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    #---------------------------------------------------------------------
    def mises_stress(self, vecStress):

        tmp1 = np.square(vecStress[0] - vecStress[1]) + np.square(vecStress[1] - vecStress[2]) + np.square(vecStress[2] - vecStress[0])
        tmp2 = 6.0 * (np.square(vecStress[3]) + np.square(vecStress[4]) + np.square(vecStress[5]))
        mises = np.sqrt(0.5 * (tmp1 + tmp2))

        return mises

