import numpy as np

# 1次元2節点トラス要素のクラス
class D1T2:

    # コンストラクタ
    # no         : 要素番号
    # nodes      : 節点のリスト(Node1d型のリスト)
    # material   : 材料特性(Material型)
    def __init__(self, no, nodes, material, area):

        # インスタンス変数を定義する
        self.no = no                  # 要素番号
        self.nodes = nodes            # 節点のリスト(Node1d型のリスト形式)
        self.mat = material           # 材料データ(Material型)
        self.area = area              # 断面積
        self.ipNum = 1                # 積分点の数
        self.w = 2.0                  # 積分点の重み係数
        self.pStrain = 0.0            # 要素内の塑性ひずみ
        self.stress = 0.0             # 要素内の応力

        # Bマトリクスを計算する
        self.matB = self.makeBmatrix()

    # 要素接線剛性マトリクスKetを作成する
    def makeKetmatrix(self):

        # dxdaを計算する
        dxda = -0.5 * self.nodes[0].x + 0.5 * self.nodes[1].x

        # 降伏状態を判定し、Dを作成する
        #if self.yeildFlg == False:
        #    D = self.mat.young
        #else:
        #    hDash = self.mat.makePlasticModule(self.pStrain)
        #    D = self.mat.young * hDash / (self.mat.young + hDash)

        # 要素接線剛性マトリクスKetを計算する
        matKet = self.area * self.w * self.matB.T * self.mat.D * self.matB * dxda

        return matKet

    # Bマトリクスを作成する
    def makeBmatrix(self):

        # Bマトリクスを計算する
        dN1dx = -1 / (-self.nodes[0].x + self.nodes[1].x)
        dN2dx = 1 / (-self.nodes[0].x + self.nodes[1].x)
        matB = np.matrix([[dN1dx, dN2dx]])

        return matB

    # Return Mapping法により、応力、塑性ひずみ、降伏判定を更新する
    # vecElemDisp : 要素節点の変位ベクトル(np.array型)
    def returnMapping(self, vecElemDisp):

        # 全ひずみを求める
        strain = np.array(self.matB @ vecElemDisp).flatten()

        # リターンマッピング法により、応力、塑性ひずみ、降伏判定を求める
        stress, pStrain = self.mat.returnMapping(strain, self.pStrain)
        self.stress = stress
        self.pStrain = pStrain

    # 内力ベクトルqを作成する
    def makeqVector(self):

        # dxdaを計算する
        dxda = -0.5 * self.nodes[0].x + 0.5 * self.nodes[1].x

        # 内力ベクトルqを計算する
        vecq = np.array(self.area * self.w * self.matB.T @ self.stress * dxda).flatten()

        return vecq

