#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

import sys
sys.path.append('../../')
import numpy as np
import numpy.linalg as LA
from material.dmatrix import Dmatrix
from model.element.element_output_data import ElementOutputData

# 6面体8節点要素のクラス
class C3D8:
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 要素(Node型のリスト)
    # material        : 構成則
    def __init__(self, no, nodes, material):

        # インスタンス変数を定義する
        self.nodeNum = 8                       # 節点の数
        self.nodeDof = 3                       # 節点の自由度
        self.no = no                           # 要素番号
        self.nodes = nodes                     # 節点の集合(Node型のリスト)
        self.material = []                     # 材料モデルのリスト

        self.ipNum = 8                         # 積分点の数
        self.w1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数1
        self.w2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数2
        self.w3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数3
        self.ai = np.array([-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),     # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0)])
        self.bi = np.array([-np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0),     # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)])
        self.ci = np.array([-np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),   # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)])
        
        #self.incNo = 0   # インクリメントのNo

        # 要素内の変位を初期化する
        self.vecDisp = np.zeros(self.nodeNum * self.nodeDof)   # 要素内の変位

        # 材料モデルを初期化する
        for ip in range(self.ipNum):
            self.material.append(material)

    # 要素接線剛性マトリクスKetを作成する
    def makeKetmatrix(self):

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # Bbarマトリクスを計算する
        matBbar = []
        for i in range(self.ipNum):
            matBbar.append(self.makeBbarmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # Ketマトリクスをガウス積分で計算する
        matKet = np.zeros([self.nodeDof * self.nodeNum, self.nodeDof * self.nodeNum])
        for i in range(self.ipNum):
            matKet += self.w1[i] * self.w2[i] * self.w3[i] * matBbar[i].T @ self.material[i].matD @ matBbar[i] * LA.det(matJ[i])

        return matKet

    # ヤコビ行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    def makeJmatrix(self, a, b, c):

         # dNdabを計算する
        matdNdabc = self.makedNdabc(a, b, c)

        # xi, yi, ziの行列を計算する
        matxiyizi = np.array([[self.nodes[0].x, self.nodes[0].y, self.nodes[0].z],
                              [self.nodes[1].x, self.nodes[1].y, self.nodes[1].z],
                              [self.nodes[2].x, self.nodes[2].y, self.nodes[2].z],
                              [self.nodes[3].x, self.nodes[3].y, self.nodes[3].z],
                              [self.nodes[4].x, self.nodes[4].y, self.nodes[4].z],
                              [self.nodes[5].x, self.nodes[5].y, self.nodes[5].z],
                              [self.nodes[6].x, self.nodes[6].y, self.nodes[6].z],
                              [self.nodes[7].x, self.nodes[7].y, self.nodes[7].z]])

        # ヤコビ行列を計算する
        matJ = matdNdabc @ matxiyizi

        # ヤコビアンが負にならないかチェックする
        if LA.det(matJ) < 0:
            raise ValueError("要素の計算に失敗しました")

        return matJ

    # Bマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    def makeBmatrix(self, a, b, c):

         # dNdabcの行列を計算する
        matdNdabc = self.makedNdabc(a, b, c)

         # ヤコビ行列を計算する
        matJ = self.makeJmatrix(a, b, c)

        # matdNdxyz = matJinv * matdNdabc
        matdNdxyz = LA.solve(matJ, matdNdabc)

        # Bマトリクスを計算する
        matB = np.empty((6,0))
        for i in range(self.nodeNum): 
            matTmp = np.array([[matdNdxyz[0, i], 0.0, 0.0],
                               [0.0, matdNdxyz[1, i], 0.0],
                               [0.0, 0.0, matdNdxyz[2, i]],
                               [0.0, matdNdxyz[2, i], matdNdxyz[1, i]],
                               [matdNdxyz[2, i], 0.0, matdNdxyz[0, i]], 
                               [matdNdxyz[1, i], matdNdxyz[0, i], 0.0]]) 
            matB = np.hstack((matB, matTmp))

        return matB

    # Bbarマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    def makeBbarmatrix(self, a, b, c):

        # Bマトリクスを作成する
        matB = self.makeBmatrix(a, b, c)

        # Bvマトリクスを作成する
        matBv = self.makeBvmatrix(a, b, c)

        # Bvbarマトリクスを作成する
        matBvbar = self.makeBvbarmatrix()

        # Bbarマトリクスを計算する
        matBbar = matBvbar + matB - matBv

        return matBbar

    # Bvマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    def makeBvmatrix(self, a, b, c):

         # dNdabcの行列を計算する
        matdNdabc = self.makedNdabc(a, b, c)

         # ヤコビ行列を計算する
        matJ = self.makeJmatrix(a, b, c)

        # matdNdxyz = matJinv * matdNdabc
        matdNdxyz = LA.solve(matJ, matdNdabc)

        # Bvマトリクスを計算する
        matBv = np.empty((6,0))
        for i in range(self.nodeNum):
            matTmp = np.array([[matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]) 
            matBv = np.hstack((matBv, matTmp))
        matBv *= 1.0 / 3.0

        return matBv

    # Bvbarマトリクスを作成する
    def makeBvbarmatrix(self):

        # 体積を計算する
        v = self.getVolume()

        # Bvマトリクスを計算する
        matBv = []
        for i in range(self.ipNum):
            matBv.append(self.makeBvmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # ガウス積分でBvbarマトリクスを計算する
        Bvbar = np.zeros([6, self.nodeNum * self.nodeDof])
        for i in range(self.ipNum):
            Bvbar += self.w1[i] * self.w2[i] * self.w3[i] * matBv[i] * LA.det(matJ[i])
        Bvbar *= 1.0 / v

        return Bvbar

    # dNdabcの行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    def makedNdabc(self, a, b, c):

        # dNi/da, dNi/db, dNi/dcを計算する
        dN1da = -0.125 * (1.0 - b) * (1.0 - c)
        dN2da = 0.125 * (1.0 - b) * (1.0 - c)
        dN3da = 0.125 * (1.0 + b) * (1.0 - c)
        dN4da = -0.125 * (1.0 + b) * (1.0 - c)
        dN5da = -0.125 * (1.0 - b) * (1.0 + c)
        dN6da = 0.125 * (1.0 - b) * (1.0 + c)
        dN7da = 0.125 * (1.0 + b) * (1.0 + c)
        dN8da = -0.125 * (1.0 + b) * (1.0 + c)
        dN1db = -0.125 * (1.0 - a) * (1.0 - c)
        dN2db = -0.125 * (1.0 + a) * (1.0 - c)
        dN3db = 0.125 * (1.0 + a) * (1.0 - c)
        dN4db = 0.125 * (1.0 - a) * (1.0 - c)
        dN5db = -0.125 * (1.0 - a) * (1.0 + c)
        dN6db = -0.125 * (1.0 + a) * (1.0 + c)
        dN7db = 0.125 * (1.0 + a) * (1.0 + c)
        dN8db = 0.125 * (1.0 - a) * (1.0 + c)
        dN1dc = -0.125 * (1.0 - a) * (1.0 - b)
        dN2dc = -0.125 * (1.0 + a) * (1.0 - b)
        dN3dc = -0.125 * (1.0 + a) * (1.0 + b)
        dN4dc = -0.125 * (1.0 - a) * (1.0 + b)
        dN5dc = 0.125 * (1.0 - a) * (1.0 - b)
        dN6dc = 0.125 * (1.0 + a) * (1.0 - b)
        dN7dc = 0.125 * (1.0 + a) * (1.0 + b)
        dN8dc = 0.125 * (1.0 - a) * (1.0 + b)

        # dNdabcを計算する
        dNdabc = np.array([[dN1da, dN2da, dN3da, dN4da, dN5da, dN6da, dN7da, dN8da],
                           [dN1db, dN2db, dN3db, dN4db, dN5db, dN6db, dN7db, dN8db],
                           [dN1dc, dN2dc, dN3dc, dN4dc, dN5dc, dN6dc, dN7dc, dN8dc]])

        return dNdabc

    # 体積を求める
    def getVolume(self):

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # ガウス積分で体積を計算する
        volume = 0
        for i in range(self.ipNum):
            volume += self.w1[i] * self.w2[i] * self.w3[i] * LA.det(matJ[i])

        return volume


    # 要素内の変数を更新する
    # vecDisp : 要素節点の変位ベクトル(np.array型)
    # incNo   : インクリメントNo
    def update(self, vecDisp, incNo):
        
        # 積分点ループ
        for i in range(self.ipNum):
            
            # Bマトリックスを作成
            matBbar = self.makeBbarmatrix(self.ai[i], self.bi[i], self.ci[i])
            
            # 構成則の内部変数の更新
            self.material[i].update(matBbar, vecDisp, incNo)
        
        # 要素内変位の更新
        self.vecDisp = vecDisp
        #self.incNo = incNo


    # 内力ベクトルqを作成する
    def makeqVector(self):

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.makeJmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # Bbarマトリクスを計算する
        matBbar = []
        for i in range(self.ipNum):
            matBbar.append(self.makeBbarmatrix(self.ai[i], self.bi[i], self.ci[i]))

        # 内力ベクトルqを計算する
        vecq = np.zeros(self.nodeDof * self.nodeNum)
        for i in range(self.ipNum):
            vecq += self.w1[i] * self.w2[i] * self.w3[i] * matBbar[i].T @ self.material[i].vecStressList * LA.det(matJ[i])

        return vecq

    # 要素の出力データを作成する
    def makeOutputData(self):

        a = 1
        #output = ElementOutputData(self,
        #                           self.vecStressList,
        #                           self.vecEStrainList,
        #                           self.vecPStrainList,
        #                           self.ePStrainList,
        #                           self.misesList)

        #return output
