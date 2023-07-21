#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import numpy as np

# 境界条件を格納するクラス
class Boundary:
    # コンストラクタ
    # num_node : 節点数
    def __init__(self, num_node):
        # インスタンス変数を定義する
        self.num_node = num_node                                     # 全節点数
        self.num_dof_at_node = 3                                           # 節点の自由度
        self.solution = np.array(num_node * self.num_dof_at_node * [None])   # 単点拘束の強制変位
        self.F = np.array(num_node * self.num_dof_at_node * [0.0])   # 荷重ベクトル

        self.matC = np.empty((0, num_node * self.num_dof_at_node))          # 多点拘束用のCマトリクス
        self.vecd = np.empty(0)                                    # 多点拘束用のdベクトル

    # 単点拘束を追加する
    # nodeNo : 節点番号
    # dispX  : x方向の強制変位
    # dispY  : y方向の強制変位
    # dispZ  : z方向の強制変位
    def add_SPC(self, nodeNo, dispX, dispY, dispZ):

        self.solution[self.num_dof_at_node * (nodeNo - 1) + 0] = dispX
        self.solution[self.num_dof_at_node * (nodeNo - 1) + 1] = dispY
        self.solution[self.num_dof_at_node * (nodeNo - 1) + 2] = dispZ

    # 多点拘束を追加する
    # 条件式 : vecC x u = d
    def addMPC(self, vecC, d):

        self.matC = np.vstack((self.matC, vecC))
        self.vecd = np.hstack((self.vecd, d))

    # 単点拘束条件から変位ベクトルを作成する
    def make_disp_vector(self):
        return self.solution

    # 荷重を追加する
    def add_force(self, nodeNo, fx, fy, fz):

        self.F[self.num_dof_at_node * (nodeNo - 1) + 0] = fx
        self.F[self.num_dof_at_node * (nodeNo - 1) + 1] = fy
        self.F[self.num_dof_at_node * (nodeNo - 1) + 2] = fz

    # 境界条件から荷重ベクトルを作成する
    def make_force_vector(self):
        return self.F
    
    # 多点拘束の境界条件を表すCマトリクス、dベクトルを作成する
    def makeMPCmatrixes(self):
        return self.matC, self.vecd
    
    # 拘束条件を出力する
    def print_boundary(self):
        print("Node Number: ", self.num_node)
        print("SPC Constraint Condition")
        print(self.solution)
        print("Force Condition")
        print(self.F)
        print("MPC Constraint Condition")
        print("C x u = d")
        print("C Matrix")
        print(self.matC)
        print("d vector")
        print(self.vecd)
