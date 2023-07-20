#https://qiita.com/Altaka4128/items/77d1f04f9b4622be617f

import numpy as np

class Boundary1d:

    # コンストラクタ
    # nodeNum : 節点数
    def __init__(self, nodeNum):

        # インスタンス変数を定義する
        self.nodeNum = nodeNum  # 全節点数
        self.nodeDof = 1                                           # 節点の自由度

        self.physical_field = np.array(nodeNum * self.nodeDof * [None])   # 単点拘束の強制変位
        self.vecForce = np.array(nodeNum * self.nodeDof * [0.0])   # 荷重ベクトル
        
        self.matC = np.empty((0, self.nodeNum))   # 多点拘束用のCマトリクス
        self.vecd = np.empty(0)                   # 多点拘束用のdベクトル

    # 単点拘束を追加する
    # nodeNo : 節点番号
    # dispX  : 強制変位量
    def add_SPC(self, nodeNo, dispX):
        self.physical_field[self.nodeDof * (nodeNo - 1) + 0] = dispX

    # 単点拘束条件から変位ベクトルを作成する
    def make_disp_vector(self):
        return self.physical_field

    # 荷重を追加する
    # nodeNo : 節点番号
    # fx     : 荷重値
    def add_force(self, nodeNo, fx):
        self.vecForce[self.nodeDof * (nodeNo - 1) + 0] = fx

    # 境界条件から荷重ベクトルを作成する
    def make_force_vector(self):
        return self.vecForce

    # 拘束条件を出力する
    def print_boundary(self):
        print("Node Number: ", self.nodeNum)
        print("SPC Constraint Condition")
        for i in range(len(self.dispNodeNo)):
            print("Node No: " + str(self.dispNodeNo[i]) + ", x: " + str(self.dispX[i]))
        print("Force Condition")
        for i in range(len(self.forceNodeNo)):
            print("Node No: " + str(self.forceNodeNo[i]) + ", x: " + str(self.forceX[i]))
