#https://qiita.com/Altaka4128/items/77d1f04f9b4622be617f

import numpy as np

class Boundary1d:

    # コンストラクタ
    # num_node : 節点数
    def __init__(self, num_node):

        # インスタンス変数を定義する
        self.num_node = num_node                                                   # 全節点数
        self.num_dof_at_node = 1                                                   # 節点の自由度

        self.physical_field = np.array(num_node * self.num_dof_at_node * [None])   # 単点拘束の強制変位
        self.F = np.array(num_node * self.num_dof_at_node * [0.0])                 # 荷重ベクトル
        
        self.matC = np.empty((0, self.num_node))   # 多点拘束用のCマトリクス
        self.vecd = np.empty(0)                    # 多点拘束用のdベクトル

    # 単点拘束を追加する
    # nodeNo : 節点番号
    # dispX  : 強制変位量
    def add_SPC(self, nodeNo, dispX):
        self.physical_field[self.num_dof_at_node * (nodeNo - 1) + 0] = dispX

    # 単点拘束条件から変位ベクトルを作成する
    def make_disp_vector(self):
        return self.physical_field

    # 荷重を追加する
    # nodeNo : 節点番号
    # fx     : 荷重値
    def add_force(self, nodeNo, fx):
        self.F[self.num_dof_at_node * (nodeNo - 1) + 0] = fx

    # 境界条件から荷重ベクトルを作成する
    def make_force_vector(self):
        return self.F
