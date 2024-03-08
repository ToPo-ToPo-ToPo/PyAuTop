#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np
import jax
import jax.numpy as jnp

# Dマトリクスを計算するためのクラス
class Dmatrix:
    # コンストラクタ
    # young   : ヤング率
    # poisson : ポアソン比
    def __init__(self, young, poisson):
        self.young = young
        self.poisson = poisson

    # 弾性状態のDマトリクスを作成する
    def make_De_matrix(self, material):
        young = material.young
        poisson = material.poisson
        tmp = young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        matD = jnp.array([[1.0 - poisson, poisson, poisson, 0.0, 0.0, 0.0],
                         [poisson, 1.0 - poisson, poisson, 0.0, 0.0, 0.0],
                         [poisson, poisson, 1.0 - poisson, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson), 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson)]])
        matD = tmp * matD

        return matD
