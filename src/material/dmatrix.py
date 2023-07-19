#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np

# Dマトリクスを計算するためのクラス
class Dmatrix:
    # コンストラクタ
    # young   : ヤング率
    # poisson : ポアソン比
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson

    # 弾性状態のDマトリクスを作成する
    def makeDematrix(self):

        tmp = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        matD = np.array([[1.0 - self.poisson, self.poisson, self.poisson, 0.0, 0.0, 0.0],
                         [self.poisson, 1.0 - self.poisson, self.poisson, 0.0, 0.0, 0.0],
                         [self.poisson, self.poisson, 1.0 - self.poisson, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson)]])
        matD = tmp * matD

        return matD
