# kato

import numpy as np
from numba import jit, f8
from numba.experimental import jitclass
spec = [
    ('young', f8),               
    ('poisson', f8),        
]
#=============================================================================
# 平面応力用のDマトリクスを計算するためのクラス
#=============================================================================
#@jitclass(spec)
class DmatrixPlaneStress:
    # コンストラクタ
    # young   : ヤング率
    # poisson : ポアソン比
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson

    #--------------------------------------------------------------
    # 弾性状態のDマトリクスを作成する
    #--------------------------------------------------------------
    #@jit(nopython=True, cache=True)
    def make_De_matrix(self):
        tmp = self.young / (1 - self.poisson * self.poisson)
        matD = np.array(
            [
                [1.0, - self.poisson, 0.0],
                [- self.poisson, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.poisson)]
            ]
        )

        matD = tmp * matD

        return matD
    
    #--------------------------------------------------------------
    # 弾性状態のC0マトリクスを作成する
    #--------------------------------------------------------------
    #@jit(nopython=True, cache=True)
    def make_C0_matrix(self):
        tmp = 1 / (1 - self.poisson * self.poisson)
        matC0 = np.array(
            [
                [1.0, self.poisson, 0.0],
                [self.poisson, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.poisson)]
            ]
        )

        matC0 = tmp * matC0

        return matC0
