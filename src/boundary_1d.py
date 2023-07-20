import numpy as np

class Boundary1d:

    # コンストラクタ
    # nodeNum : 節点数
    def __init__(self, nodeNum):

        # インスタンス変数を定義する
        self.nodeNum = nodeNum  # 全節点数
        self.dispNodeNo = []    # 単点拘束の節点番号
        self.dispX = []         # 単点拘束の強制変位x
        self.forceNodeNo = []   # 荷重が作用する節点番号
        self.forceX = []        # x方向の荷重
        self.matC = np.empty((0, self.nodeNum))   # 多点拘束用のCマトリクス
        self.vecd = np.empty(0)                   # 多点拘束用のdベクトル

    # 単点拘束を追加する
    # nodeNo : 節点番号
    # x      : 強制変位量
    def addSPC(self, nodeNo, x):

        self.dispNodeNo.append(nodeNo)
        self.dispX.append(x)

    # 単点拘束条件から変位ベクトルを作成する
    def makeDispVector(self):

        vecd = np.array([None] * self.nodeNum)
        for i in range(len(self.dispNodeNo)):
            vecd[(self.dispNodeNo[i] - 1)] = self.dispX[i]

        return vecd

    # 荷重を追加する
    # nodeNo : 節点番号
    # fx     : 荷重値
    def addForce(self, nodeNo, fx):

        self.forceNodeNo.append(nodeNo)
        self.forceX.append(fx)

    # 境界条件から荷重ベクトルを作成する
    def makeForceVector(self):

        vecf = np.array(np.zeros([self.nodeNum]))
        for i in range(len(self.forceNodeNo)):
            vecf[(self.forceNodeNo[i] - 1)] += self.forceX[i]

        return vecf


    # 拘束条件を出力する
    def printBoundary(self):
        print("Node Number: ", self.nodeNum)
        print("SPC Constraint Condition")
        for i in range(len(self.dispNodeNo)):
            print("Node No: " + str(self.dispNodeNo[i]) + ", x: " + str(self.dispX[i]))
        print("Force Condition")
        for i in range(len(self.forceNodeNo)):
            print("Node No: " + str(self.forceNodeNo[i]) + ", x: " + str(self.forceX[i]))
