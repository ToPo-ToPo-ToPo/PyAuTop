
import numpy as np

# 材料情報を格納するクラス
class ElastoPlasticVonMisesTruss:

    # コンストラクタ
    # young    : ヤング率
    # area     : 面積
    def __init__(self, young, area):

        # インスタンス変数を定義する
        self.young = young      # ヤング率
        self.area = area        # 面積
        self.stressLine = []    # 応力-塑性ひずみ多直線の応力データ
        self.pStrainLine = []   # 応力-塑性ひずみ多直線の塑性ひずみデータ
        self.itrNum = 100       # ニュートン・ラプソン法の収束回数の上限
        self.tol = 0.001        # ニュートン・ラプソン法の収束基準

    # 応力-塑性ひずみ多直線のデータを追加する
    # (入力されるデータは塑性ひずみが小さい順になり、最初の塑性ひずみは0.0にならなければいけない)
    # (塑性係数は正になる前提)
    # stress  : 応力(正の前提)
    # pStrain : 塑性ひずみ(正の前提)
    def addStressPStrainLine(self, stress, pStrain):

        # 入力チェック
        if len(self.pStrainLine) == 0:
            if not pStrain == 0.0 :
                raise ValueError("応力-塑性ひずみ多直線データの最初のひずみは0.0になる必要があります。")
        elif self.pStrainLine[-1] > pStrain:
            raise ValueError("応力-塑性ひずみ多直線データの入力順序が間違っています。")

        # 最初の入力の場合は降伏応力を格納する
        if len(self.stressLine) == 0:
            self.yieldStress = stress   # 降伏応力

        self.stressLine.append(stress)
        self.pStrainLine.append(pStrain)

    # 降伏関数を計算する
    # (0より大きければ降伏している)
    # stress  : 応力
    # pStrain : 塑性ひずみ
    def yieldFunction(self, stress, pStrain):

        f = 0.0
        if hasattr(self, 'yieldStress'):
            f = np.abs(stress) - self.makeYeildStress(pStrain)

        return f

    # 降伏応力を求める
    # pStrain : 塑性ひずみ
    def makeYeildStress(self, pStrain):

        yeildStress = 0.0
        if hasattr(self, 'yieldStress'):
            # pStrainが何番目のデータの間か求める
            no = None
            for i in range(len(self.pStrainLine) - 1):
                if self.pStrainLine[i] <= np.abs(pStrain) and np.abs(pStrain) <= self.pStrainLine[i+1]:
                    no = i
                    break
            if no is None :
                raise ValueError("塑性ひずみが定義した範囲を超えています。")

            hDash = self.makePlasticModule(pStrain)
            yeildStress = hDash * (np.abs(pStrain) - self.pStrainLine[no]) + self.stressLine[no]

        return yeildStress

    # 塑性係数を求める
    # pStrain : 塑性ひずみ
    def makePlasticModule(self, pStrain):

        # pStrainが何番目のデータの間か求める
        no = None
        for i in range(len(self.pStrainLine) - 1):
            if self.pStrainLine[i] <= np.abs(pStrain) and np.abs(pStrain) <= self.pStrainLine[i+1]:
                no = i
                break
        if no is None :
            raise ValueError("塑性ひずみが定義した範囲を超えています。")

        # 塑性係数を計算する
        h = (self.stressLine[no+1] - self.stressLine[no]) / (self.pStrainLine[no+1] - self.pStrainLine[no])

        return h

    # Return Mapping法により、ひずみ増分、降伏判定を更新する
    # strain      : 全ひずみ
    # prevPStrain : 前回の塑性ひずみ
    def returnMapping(self, strain, prevPStrain):

        # 試行弾性応力を求める
        triStrain = strain - prevPStrain
        triStress = self.young * triStrain

        # 降伏関数を計算する
        triF = self.yieldFunction(triStress, np.abs(prevPStrain))

        # 降伏判定を計算する
        if triF > 0.0 :
            yeildFlg = True
        else:
            yeildFlg = False

        # 塑性ひずみの増分量をニュートン・ラプソン法で計算する
        deltaGamma = 0.0
        if triF > 0.0 :
            kn = self.makeYeildStress(prevPStrain)
            # 収束演算を行う
            for i in range(self.itrNum):
                # y, y'を計算する
                k = self.makeYeildStress(prevPStrain + deltaGamma)
                hDash = self.makePlasticModule(prevPStrain + deltaGamma)
                y = triF - self.young * deltaGamma - k + kn
                yDash = - self.young - hDash

                # 収束判定を行う
                if np.abs(y) < self.tol:
                    break

                deltaGamma -= y / yDash


        deltaPStrain = deltaGamma * np.sign(triStress)

        # 塑性ひずみを計算する
        pStrain = prevPStrain + deltaPStrain

        # 応力を計算する
        stress = triStress - self.young * deltaPStrain

        return stress, pStrain, yeildFlg
