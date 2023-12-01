# kato

import copy
import numpy as np
import numpy.linalg as LA
from src.physics.element.element_base import ElementBase 
#=============================================================================
# 4節点平面応力要素のクラス
#=============================================================================
class CPS4(ElementBase):
    # コンストラクタ
    # no           : 要素番号
    # nodes        : 節点の集合(Node型のリスト)
    # young        : ヤング率
    # poisson      : ポアソン比
    # density      : 密度
    # vecGravity   : 重力加速度のベクトル(np.array型)
    def __init__(self, no, nodes, material, vecGravity = None):

        # インスタンス変数を定義する
        self.num_node = 4            # 節点の数
        self.num_dof_at_node = 2     # 節点の自由度
        self.no = no                 # 要素番号
        self.nodes = nodes             # nodesは反時計回りの順番になっている前提(Node2d型のリスト形式)
        self.material = []             # 材料モデルのリスト
        self.vecGravity = vecGravity   # 重力加速度のベクトル(np.array型)
        
        self.ipNum = 4                 # 積分点の数
        self.w1 = [1.0, 1.0, 1.0, 1.0]  # 積分点の重み係数1
        self.w2 = [1.0, 1.0, 1.0, 1.0]  # 積分点の重み係数2
        self.ai = np.array([-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0)])  # 積分点の座標(a,b座標系)
        self.bi = np.array([-np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)])  # 積分点の座標(a,b座標系)


        # 要素内節点の自由度数を更新する
        for inode in range(len(self.nodes)):
            self.nodes[inode].num_dof = self.num_dof_at_node # .num_dofってなんだ??? <- nodes.pyの中にあるself.num_dofを要素の自由度数に設定している

        # 要素内の変位を初期化する
        self.solution = np.zeros(self.num_node * self.num_dof_at_node)
        
        # 要素内の自由度番号のリストを作成する
        self.dof_list = self.make_dof_list(nodes, self.num_node, self.num_dof_at_node)

        # 材料モデルを初期化する
        for ip in range(self.ipNum):
            self.material.append(copy.deepcopy(material))
    
    #---------------------------------------------------------------------
    # 要素剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    def make_K(self):

        # 初期化
        Ke = np.zeros([self.num_dof_at_node * self.num_node, self.num_dof_at_node * self.num_node])

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(ip)

            # Bマトリクスを計算する
            matB = self.make_B_matrix(ip)

            # Ketマトリクスをガウス積分で計算する
            Ke += self.w1[ip] * self.w2[ip] * matB.T @ self.material[ip].matD @ matB * LA.det(matJ)

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
            matJ = self.make_J_matrix(ip)

            # Bbarマトリクスを計算する
            matB = self.make_B_matrix(ip)

            # 内力ベクトルを計算する
            Fint_e += self.w1[ip] * self.w2[ip] * matB.T @ self.material[ip].vecStress * LA.det(matJ)

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
    #---------------------------------------------------------------------
    def make_J_matrix(self, ip):

        # dNdabを計算する
        matdNdab = self.make_dNda(ip)

        # xi, yiの行列を計算する
        matxiyi = np.array([[self.nodes[0].x, self.nodes[0].y],
                              [self.nodes[1].x, self.nodes[1].y],
                              [self.nodes[2].x, self.nodes[2].y],
                              [self.nodes[3].x, self.nodes[3].y]])

        # ヤコビ行列を計算する
        matJ = matdNdab @ matxiyi

        # ヤコビアンが負にならないかチェックする
        if LA.det(matJ) < 0:
            raise ValueError("要素の計算に失敗しました")

        return matJ

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    # a : a座標値
    # b : b座標値
    #---------------------------------------------------------------------
    def make_B_matrix(self, ip):

        # 積分点座標を設定する
        a = self.ai[ip]
        b = self.bi[ip]

        # dNdaの行列を計算する
        dNda = self.make_dNda(ip)

        # ヤコビ行列を計算する
        matJ = self.make_J_matrix(ip)

        # matdNdxy = matJinv * matdNdab
        matdNdxy = LA.solve(matJ, dNda)  # J @ dNdxy = dNdaを解いている

        # Bマトリクスを計算する
        matB = np.empty((3,0))
        for i in range(self.num_node): 
            matTmp = np.array([[matdNdxy[0, i], 0.0],
                               [0.0, matdNdxy[1, i]],
                               [matdNdxy[1, i], matdNdxy[0, i]]]) # 行列∂
            matB = np.hstack((matB, matTmp))

        return matB

    #---------------------------------------------------------------------
    # dNdxの行列を計算する
    # a : a座標値
    # b : b座標値
    #---------------------------------------------------------------------
    def make_dNda(self, ip):

        # 積分点座標を設定する
        a = self.ai[ip]
        b = self.bi[ip]

        # dNi/da, dNi/db, dNi/dcを計算する
        dN1da = -0.25 * (1.0 - b)
        dN2da = 0.25 * (1.0 - b)
        dN3da = 0.25 * (1.0 + b)
        dN4da = -0.25 * (1.0 + b)
        dN1db = -0.25 * (1.0 - a) 
        dN2db = -0.25 * (1.0 + a) 
        dN3db = 0.25 * (1.0 + a) 
        dN4db = 0.25 * (1.0 - a) 

        # dNdaを計算する
        dNda = np.array([[dN1da, dN2da, dN3da, dN4da],
                         [dN1db, dN2db, dN3db, dN4db]])

        return dNda

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
            matB = self.make_B_matrix(ip)
            
            # 構成則の内部変数の更新
            self.material[ip].compute_stress_and_tangent_matrix(matB, elem_solution)
    
    #---------------------------------------------------------------------
    # 構成則の変数を更新する
    #---------------------------------------------------------------------
    def update_constitutive_law(self):

        # 積分点ループ
        for ip in range(self.ipNum):
            self.material[ip].update()
