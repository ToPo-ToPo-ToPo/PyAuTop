# kato

import numpy as np

# 平面応力用のDマトリクスを計算するためのクラス
class DmatrixPlaneStress:
    # コンストラクタ
    # young   : ヤング率
    # poisson : ポアソン比
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson

    # 弾性状態のDマトリクスを作成する
    def make_De_matrix(self):
        tmp = self.young / (1 - self.poisson * self.poisson)
        # matD = np.array(
        #     [
        #         [1.0, - self.poisson, 0.0, 0.0, 0.0, 0.0],
        #         [- self.poisson, 1.0, 0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.5 * (1.0 - self.poisson), 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #     ]
        # )
        matD = np.array(
            [
                [1.0, - self.poisson, 0.0],
                [- self.poisson, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.poisson)]
            ]
        )

        matD = tmp * matD

        return matD
