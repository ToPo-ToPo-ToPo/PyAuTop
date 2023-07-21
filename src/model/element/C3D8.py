#https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import copy
import numpy as np
import numpy.linalg as LA
#=============================================================================
# 6面体8節点要素のクラス
#=============================================================================
class C3D8:
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 要素(Node型のリスト)
    # material        : 構成則
    def __init__(self, no, nodes, material):

        # インスタンス変数を定義する
        self.num_node = 8                      # 節点の数
        self.num_dof_at_node = 3               # 節点の自由度
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
        
        # 要素内の変位を初期化する
        self.solution = np.zeros(self.num_node * self.num_dof_at_node)   # 要素内の変位

        # 材料モデルを初期化する
        for ip in range(self.ipNum):
            self.material.append(copy.deepcopy(material))

    #---------------------------------------------------------------------
    # 要素接線剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    def make_K(self):

        # 初期化
        Ke = np.zeros([self.num_dof_at_node * self.num_node, self.num_dof_at_node * self.num_node])

        # 積分点ループ
        for ip in range(self.ipNum):

            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # Bbarマトリクスを計算する
            matBbar = self.make_Bbar_matrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # 要素剛性行列Keを計算する
            Ke += self.w1[ip] * self.w2[ip] * self.w3[ip] * matBbar.T @ self.material[ip].matD @ matBbar * LA.det(matJ)

        return Ke
    
    #---------------------------------------------------------------------
    # 内力ベクトルFintを作成する
    #---------------------------------------------------------------------
    def make_Fint(self):

        # 初期化
        Fint_e = np.zeros(self.num_dof_at_node * self.num_node)

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # Bbarマトリクスを計算する
            matBbar = self.make_Bbar_matrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # 内力ベクトルを計算する
            Fint_e += self.w1[ip] * self.w2[ip] * self.w3[ip] * matBbar.T @ self.material[ip].vecStress * LA.det(matJ)

        return Fint_e
    
    #---------------------------------------------------------------------
    # 等価節点力の荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fb(self):

        # 初期化
        Fb = np.zeros(self.num_node * self.num_dof_at_node)

        # 積分点ループ
        '''for ip in range(self.ipNum):
            
            # ヤコビ行列を計算する
            matJ = self.makeJmatrix(self.ai[ip], self.bi[ip], self.ci[ip])

            # 物体力による等価節点力を計算する
            if not self.vecGravity is None:
                vecb = self.material[ip].density * self.vecGravity   # 単位体積あたりの物体力のベクトル
                N1 = 1 - self.ai - self.bi - self.ci
                N2 = self.ai
                N3 = self.bi
                N4 = self.ci
                
                matN = np.matrix([[N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0, 0.0],
                                  [0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0],
                                  [0.0, 0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4]])
                
                Fb += self.w * np.array(matN.T @ vecb).flatten() * LA.det(matJ)'''
                
        return Fb
    
    #---------------------------------------------------------------------
    # ヤコビ行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_J_matrix(self, a, b, c):

         # dNdabを計算する
        matdNdabc = self.make_dNda(a, b, c)

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

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_B_matrix(self, a, b, c):

        # dNdaの行列を計算する
        dNda = self.make_dNda(a, b, c)

        # ヤコビ行列を計算する
        matJ = self.make_J_matrix(a, b, c)

        # matdNdxyz = matJinv * matdNdabc
        matdNdxyz = LA.solve(matJ, dNda)

        # Bマトリクスを計算する
        matB = np.empty((6,0))
        for i in range(self.num_node): 
            matTmp = np.array([[matdNdxyz[0, i], 0.0, 0.0],
                               [0.0, matdNdxyz[1, i], 0.0],
                               [0.0, 0.0, matdNdxyz[2, i]],
                               [0.0, matdNdxyz[2, i], matdNdxyz[1, i]],
                               [matdNdxyz[2, i], 0.0, matdNdxyz[0, i]], 
                               [matdNdxyz[1, i], matdNdxyz[0, i], 0.0]]) 
            matB = np.hstack((matB, matTmp))

        return matB

    #---------------------------------------------------------------------
    # Bbarマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_Bbar_matrix(self, a, b, c):

        # Bマトリクスを作成する
        matB = self.make_B_matrix(a, b, c)

        # Bvマトリクスを作成する
        matBv = self.make_Bv_matrix(a, b, c)

        # Bvbarマトリクスを作成する
        matBvbar = self.make_Bvbar_matrix()

        # Bbarマトリクスを計算する
        matBbar = matBvbar + matB - matBv

        return matBbar

    #---------------------------------------------------------------------
    # Bvマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_Bv_matrix(self, a, b, c):

         # dNdabcの行列を計算する
        matdNdabc = self.make_dNda(a, b, c)

         # ヤコビ行列を計算する
        matJ = self.make_J_matrix(a, b, c)

        # matdNdxyz = matJinv * matdNdabc
        matdNdxyz = LA.solve(matJ, matdNdabc)

        # Bvマトリクスを計算する
        matBv = np.empty((6,0))
        for i in range(self.num_node):
            matTmp = np.array([[matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [matdNdxyz[0, i], matdNdxyz[1, i], matdNdxyz[2, i]],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]) 
            matBv = np.hstack((matBv, matTmp))
        matBv *= 1.0 / 3.0

        return matBv

    #---------------------------------------------------------------------
    # Bvbarマトリクスを作成する
    #---------------------------------------------------------------------
    def make_Bvbar_matrix(self):

        # 体積を計算する
        v = self.get_volume()

        # Bvマトリクスを計算する
        matBv = []
        for i in range(self.ipNum):
            matBv.append(self.make_Bv_matrix(self.ai[i], self.bi[i], self.ci[i]))

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.make_J_matrix(self.ai[i], self.bi[i], self.ci[i]))

        # ガウス積分でBvbarマトリクスを計算する
        Bvbar = np.zeros([6, self.num_node * self.num_dof_at_node])
        for i in range(self.ipNum):
            Bvbar += self.w1[i] * self.w2[i] * self.w3[i] * matBv[i] * LA.det(matJ[i])
        Bvbar *= 1.0 / v

        return Bvbar

    #---------------------------------------------------------------------
    # dNdxの行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_dNda(self, a, b, c):

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

        # dNdaを計算する
        dNda = np.array([[dN1da, dN2da, dN3da, dN4da, dN5da, dN6da, dN7da, dN8da],
                         [dN1db, dN2db, dN3db, dN4db, dN5db, dN6db, dN7db, dN8db],
                         [dN1dc, dN2dc, dN3dc, dN4dc, dN5dc, dN6dc, dN7dc, dN8dc]])

        return dNda

    #---------------------------------------------------------------------
    # 体積を求める
    #---------------------------------------------------------------------
    def get_volume(self):

        # ヤコビ行列を計算する
        matJ = []
        for i in range(self.ipNum):
            matJ.append(self.make_J_matrix(self.ai[i], self.bi[i], self.ci[i]))

        # ガウス積分で体積を計算する
        volume = 0
        for i in range(self.ipNum):
            volume += self.w1[i] * self.w2[i] * self.w3[i] * LA.det(matJ[i])

        return volume

    #---------------------------------------------------------------------
    # 構成則の計算を行う
    # elem_solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def compute_constitutive_law(self, elem_solution):
        
        # 要素内変位の更新
        self.solution = elem_solution

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # Bマトリックスを作成
            matBbar = self.make_Bbar_matrix(self.ai[ip], self.bi[ip], self.ci[ip])
            
            # 構成則の内部変数の更新
            self.material[ip].compute_stress_and_tangent_matrix(matBbar, elem_solution)
    
    #---------------------------------------------------------------------
    # 構成則の変数を更新する
    #---------------------------------------------------------------------
    def update_constitutive_law(self):

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # 構成則の内部変数の更新
            self.material[ip].update()
