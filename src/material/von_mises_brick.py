#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import numpy as np
import numpy.linalg as LA
from src.material.dmatrix import Dmatrix

# Misesモデルの構成則を計算するためのクラス
class MisesMaterial:
    # コンストラクタ
    # young       : ヤング率
    # poisson     : ポアソン比
    # density     : 密度
    def __init__(self, young, poisson, density):

        # インスタンス変数を定義する
        self.young = young                               # ヤング率
        self.poisson = poisson                           # ポアソン比
        self.density = density                           # 密度
        self.G = young / (2.0 * (1.0 + poisson))         # 横弾性係数
        self.K = young / (3.0 * (1.0 - 2.0 * poisson))   # 体積弾性率
        self.stressLine = []                             # 真応力-塑性ひずみ多直線の真応力データ
        self.pStrainLine = []                            # 真応力-塑性ひずみ多直線の塑性ひずみデータ
        self.itrNum = 100                                # ニュートン・ラプソン法の収束回数の上限
        self.tol = 1.0e-06                               # ニュートン・ラプソン法の収束基準

    # 真応力-塑性ひずみ多直線のデータを追加する
    # (入力されるデータはひずみが小さい順になり、最初の塑性ひずみは0.0にならなければいけない)
    # (塑性係数は正になる前提)
    # stress  : 真応力
    # pStrain : 塑性ひずみ
    def addStressPStrainLine(self, stress, pStrain):

        # 入力チェック
        if len(self.pStrainLine) == 0:
            if not pStrain == 0.0 :
                raise ValueError("応力-ひずみ多直線データの最初のひずみは0.0になる必要があります。")
        elif self.pStrainLine[-1] > pStrain:
            raise ValueError("応力-ひずみ多直線データの入力順序が間違っています。")

        # 最初の入力の場合は降伏応力を格納する
        if len(self.stressLine) == 0:
            self.yieldStress = stress   # 降伏応力

        self.stressLine.append(stress)
        self.pStrainLine.append(pStrain)

    # 降伏関数を計算する
    # mStress  : 相当応力
    # ePStrain : 相当塑性ひずみ
    def yieldFunction(self, mStress, ePStrain):

        f = 0.0
        if hasattr(self, 'yieldStress'):
            f = mStress - self.makeYieldStress(ePStrain)

        return f

    # 降伏応力を求める
    # ePStrain : 相当塑性ひずみ
    def makeYieldStress(self, ePStrain):

        yieldStress = 0.0
        if hasattr(self, 'yieldStress'):
            # pStrainが何番目のデータの間か求める
            no = None
            for i in range(len(self.pStrainLine) - 1):
                if self.pStrainLine[i] <= ePStrain and ePStrain <= self.pStrainLine[i+1]:
                    no = i
                    break
            if no is None :
                raise ValueError("相当塑性ひずみが定義した範囲を超えています。")

            hDash = self.makePlasticModule(ePStrain)
            yieldStress = hDash * (ePStrain - self.pStrainLine[no]) + self.stressLine[no]

        return yieldStress


    # 塑性係数を求める
    # ePStrain : 相当塑性ひずみ
    def makePlasticModule(self, ePStrain):

        # epが何番目のデータの間か求める
        no = None
        for i in range(len(self.pStrainLine) - 1):
            if self.pStrainLine[i] <= ePStrain and ePStrain <= self.pStrainLine[i+1]:
                no = i
        if no is None :
            raise ValueError("相当塑性ひずみが定義した範囲を超えています。")

        # 塑性係数を計算する
        h = (self.stressLine[no+1] - self.stressLine[no]) / (self.pStrainLine[no+1] - self.pStrainLine[no])

        return h

    # Return Mapping法により、3次元の応力、塑性ひずみ、相当塑性ひずみ、降伏判定を計算する
    # vecStrain      : 全ひずみ
    # vecPrevPStrain : 前回の塑性ひずみ
    # prevEPStrain   : 前回の相当塑性ひずみ
    def returnMapping3D(self, vecStrain, vecPrevPStrain, prevEPStrain):

        # 偏差ひずみのテンソルを求める
        mStrain = (1.0 / 3.0) * (vecStrain[0] + vecStrain[1] + vecStrain[2])
        tenStrain = np.array([[vecStrain[0], vecStrain[5] * 0.5, vecStrain[4] * 0.5],
                              [vecStrain[5] * 0.5, vecStrain[1], vecStrain[3] * 0.5],
                              [vecStrain[4] * 0.5, vecStrain[3] * 0.5, vecStrain[2]]])
        tenDStrain = tenStrain - mStrain * np.eye(3)

        # 前回の塑性ひずみのテンソルを求める
        tenPrevPStrain = np.array([[vecPrevPStrain[0], vecPrevPStrain[5] * 0.5, vecPrevPStrain[4] * 0.5],
                                   [vecPrevPStrain[5] * 0.5, vecPrevPStrain[1], vecPrevPStrain[3] * 0.5],
                                   [vecPrevPStrain[4] * 0.5, vecPrevPStrain[3] * 0.5, vecPrevPStrain[2]]])

        # 試行弾性偏差応力のテンソルを求める
        tenTriDStress = 2.0 * self.G * (tenDStrain - tenPrevPStrain)

        # 試行弾性応力のミーゼス応力を求める
        mTriStress = np.sqrt(3.0 / 2.0) * LA.norm(tenTriDStress, "fro")

        # 降伏関数を計算する
        triF = self.yieldFunction(mTriStress, prevEPStrain)

        # 降伏判定を計算する
        if triF > 0.0:
            yieldFlg = True
        else:
            yieldFlg = False

        # ΔGammaをニュートン・ラプソン法で計算する
        deltaGamma = 0.0
        if triF > 0.0 :
            normTriDStress = LA.norm(tenTriDStress, "fro")
            # 収束演算を行う
            for i in range(self.itrNum):

                # yを計算する
                yieldStress = self.makeYieldStress(prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma)
                y = normTriDStress - 2.0 * self.G * deltaGamma - np.sqrt(2.0 / 3.0) * yieldStress

                # y'を計算する
                hDash = self.makePlasticModule(prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma)
                yDash = - 2.0 * self.G - (2.0 / 3.0) * hDash

                # 収束判定を行う
                if np.abs(y) < self.tol:
                    break
                elif (i + 1) == self.itrNum:
                    raise ValueError("ニュートン・ラプソン法が収束しませんでした。") 

                # ΔGammaを更新する
                deltaGamma -= y / yDash            

        # ΔGammaの値をチェックする
        if deltaGamma < 0:
            raise ValueError("ΔGammaが負になりました。")

        # テンソルNを計算する
        tenN = tenTriDStress / LA.norm(tenTriDStress, "fro")

        # 塑性ひずみのテンソルを計算する
        tenPStrain = tenPrevPStrain + tenN * deltaGamma

        # 弾性ひずみのテンソルを計算する
        tenEStrain = tenStrain - tenPStrain

        # 応力のテンソルを計算する
        vStrain = vecStrain[0] + vecStrain[1] + vecStrain[2]
        mSigma = self.K * vStrain
        tenDStress = 2.0 * self.G * (tenDStrain - tenPStrain)
        tenStress = tenDStress + mSigma * np.eye(3)

        # 相当塑性ひずみを計算する
        ePStrain = prevEPStrain + np.sqrt(2.0 / 3.0) * deltaGamma

        # 応力を求める
        vecStress = np.array([tenStress[0,0], 
                              tenStress[1,1], 
                              tenStress[2,2], 
                              tenStress[1,2],
                              tenStress[0,2],
                              tenStress[0,1]])

        # 弾性ひずみを求める
        vecEStrain = np.array([tenEStrain[0,0], 
                               tenEStrain[1,1], 
                               tenEStrain[2,2], 
                               2.0 * tenEStrain[1,2],
                               2.0 * tenEStrain[0,2],
                               2.0 * tenEStrain[0,1]])

        # 塑性ひずみを求める
        vecPStrain = np.array([tenPStrain[0,0], 
                               tenPStrain[1,1], 
                               tenPStrain[2,2], 
                               2.0 * tenPStrain[1,2],
                               2.0 * tenPStrain[0,2],
                               2.0 * tenPStrain[0,1]])

        # 整合接線係数を求める
        if yieldFlg == True:
            matDep = self.makeDepmatrix3D(vecStress, ePStrain, prevEPStrain)
        else:
            matDep = Dmatrix(self.young, self.poisson).makeDematrix()

        return vecStress, vecEStrain, vecPStrain, ePStrain, yieldFlg, matDep

    # 3次元のミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    def misesStress3D(self, vecStress):

        tmp1 = np.square(vecStress[0] - vecStress[1]) + np.square(vecStress[1] - vecStress[2]) + np.square(vecStress[2] - vecStress[0])
        tmp2 = 6.0 * (np.square(vecStress[3]) + np.square(vecStress[4]) + np.square(vecStress[5]))
        mises = np.sqrt(0.5 * (tmp1 + tmp2))

        return mises

    # 塑性状態のDマトリクスを作成する
    # vecStress    : 応力
    # ePStrain     : 相当塑性ひずみ
    # prevEPStrain : 前の相当塑性ひずみ 
    def makeDepmatrix3D(self, vecStress, ePStrain, prevEPStrain):

        # Deマトリクスを計算する
        matDe = Dmatrix(self.young, self.poisson).makeDematrix()

        # gammmaDashを計算する
        deltaEPStrain = ePStrain - prevEPStrain
        mStress = self.misesStress3D(vecStress)
        gammaDash = 3.0 * deltaEPStrain / (2.0 * mStress)

        matP = (1.0 / 3.0) * np.array([[2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                                       [-1.0, 2.0, -1.0, 0.0, 0.0, 0.0],
                                       [-1.0, -1.0, 2.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]])
        matA = LA.inv(LA.inv(matDe) + gammaDash * matP)
        hDash = self.makePlasticModule(ePStrain)
        a = np.power(1.0 - (2.0 / 3.0) * gammaDash * hDash, -1)
        vecDStress = matP @ vecStress
        tmp1 = np.array(matA @ (np.matrix(vecDStress).T * np.matrix(vecDStress)) @ matA)
        tmp2 = (4.0 / 9.0) * a * hDash * mStress ** 2 + (np.matrix(vecDStress) @ matA @ np.matrix(vecDStress).T)[0,0]
        matDep = np.array(matA - tmp1 / tmp2)

        return matDep
