#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np

# 境界条件を格納するクラス
class Boundary:
    # コンストラクタ
    # nodeNum : 節点数
    def __init__(self, nodeNum):
        # インスタンス変数を定義する
        self.nodeNum = nodeNum                                     # 全節点数
        self.nodeDof = 3                                           # 節点の自由度
        self.physical_field = np.array(nodeNum * self.nodeDof * [None])   # 単点拘束の強制変位
        self.vecForce = np.array(nodeNum * self.nodeDof * [0.0])   # 荷重ベクトル

        self.matC = np.empty((0, nodeNum * self.nodeDof))          # 多点拘束用のCマトリクス
        self.vecd = np.empty(0)                                    # 多点拘束用のdベクトル

    # 単点拘束を追加する
    # nodeNo : 節点番号
    # dispX  : x方向の強制変位
    # dispY  : y方向の強制変位
    # dispZ  : z方向の強制変位
    def add_SPC(self, nodeNo, dispX, dispY, dispZ):

        self.physical_field[self.nodeDof * (nodeNo - 1) + 0] = dispX
        self.physical_field[self.nodeDof * (nodeNo - 1) + 1] = dispY
        self.physical_field[self.nodeDof * (nodeNo - 1) + 2] = dispZ

    # 多点拘束を追加する
    # 条件式 : vecC x u = d
    def addMPC(self, vecC, d):

        self.matC = np.vstack((self.matC, vecC))
        self.vecd = np.hstack((self.vecd, d))

    # 単点拘束条件から変位ベクトルを作成する
    def make_disp_vector(self):
        return self.physical_field

    # 荷重を追加する
    def add_force(self, nodeNo, fx, fy, fz):

        self.vecForce[self.nodeDof * (nodeNo - 1) + 0] = fx
        self.vecForce[self.nodeDof * (nodeNo - 1) + 1] = fy
        self.vecForce[self.nodeDof * (nodeNo - 1) + 2] = fz

    # 境界条件から荷重ベクトルを作成する
    def make_force_vector(self):
        return self.vecForce
    
    # 多点拘束の境界条件を表すCマトリクス、dベクトルを作成する
    def makeMPCmatrixes(self):
        return self.matC, self.vecd
    
    # 拘束条件を出力する
    def print_boundary(self):
        print("Node Number: ", self.nodeNum)
        print("SPC Constraint Condition")
        print(self.physical_field)
        print("Force Condition")
        print(self.vecForce)
        print("MPC Constraint Condition")
        print("C x u = d")
        print("C Matrix")
        print(self.matC)
        print("d vector")
        print(self.vecd)
