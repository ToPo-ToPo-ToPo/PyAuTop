#https://qiita.com/Altaka4128/items/41101c96729b68d7c96f

from os.path import dirname, abspath
import sys
parent_dir = dirname(dirname(dirname(abspath(__file__))))
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

import copy
import numpy as np
import numpy.linalg as LA
#=============================================================================
# 四面体4節点要素のクラス
#=============================================================================
class C3D4:
    # コンストラクタ
    # no           : 要素番号
    # nodes        : 節点の集合(Node型のリスト)
    # young        : ヤング率
    # poisson      : ポアソン比
    # density      : 密度
    # vecGravity   : 重力加速度のベクトル(np.array型)
    def __init__(self, no, nodes, material, vecGravity = None):

        # インスタンス変数を定義する
        self.num_node = 4              # 節点の数
        self.num_dof_at_node = 3       # 節点の自由度
        self.no = no                   # 要素番号
        self.nodes = nodes             # nodesは反時計回りの順番になっている前提(Node2d型のリスト形式)
        self.material = []             # 材料モデルのリスト
        self.vecGravity = vecGravity   # 重力加速度のベクトル(np.array型)
        
        self.ipNum = 1                 # 積分点の数
        self.w = 1.0 / 6.0             # 積分点の重み係数
        self.ai = 1.0 / 4.0            # 積分点の座標(a,b,c座標系)
        self.bi = 1.0 / 4.0            # 積分点の座標(a,b,c座標系)
        self.ci = 1.0 / 4.0            # 積分点の座標(a,b,c座標系)

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
            matJ = self.make_J_matrix()

            # Bマトリクスを計算する
            matB = self.make_B_matrix()

            # Ketマトリクスをガウス積分で計算する
            Ke += self.w * matB.T @ self.material[ip].matD @ matB * LA.det(matJ)

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
            matJ = self.make_J_matrix()

            # Bbarマトリクスを計算する
            matB = self.make_B_matrix()

            # 内力ベクトルを計算する
            Fint_e += self.w * matB.T @ self.material[ip].vecStress * LA.det(matJ)

        return Fint_e

    #---------------------------------------------------------------------
    # 等価節点力の荷重ベクトルを作成する
    #---------------------------------------------------------------------
    def make_Fb(self):

        # ヤコビ行列を計算する
        matJ = self.make_J_matrix()

        # 初期化
        Fb = np.zeros(self.num_node * self.num_dof_at_node)

        # 物体力による等価節点力を計算する
        if not self.vecGravity is None:

            # 積分点ループ
            for ip in range(self.ipNum):
                # 単位体積あたりの物体力のベクトル
                vecb = self.material[ip].density * self.vecGravity   
                N1 = 1 - self.ai - self.bi - self.ci
                N2 = self.ai
                N3 = self.bi
                N4 = self.ci
                matN = np.matrix([[N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0, 0.0],
                                  [0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4, 0.0],
                                  [0.0, 0.0, N1, 0.0, 0.0, N2, 0.0, 0.0, N3, 0.0, 0.0, N4]])
                
                Fb += self.w * np.array(matN.T @ vecb).flatten() * LA.det(matJ)

        return Fb
    
    #---------------------------------------------------------------------
    # ヤコビ行列を計算する
    #---------------------------------------------------------------------
    def make_J_matrix(self):

        dxda = -self.nodes[0].x + self.nodes[1].x
        dyda = -self.nodes[0].y + self.nodes[1].y
        dzda = -self.nodes[0].z + self.nodes[1].z
        dxdb = -self.nodes[0].x + self.nodes[2].x
        dydb = -self.nodes[0].y + self.nodes[2].y
        dzdb = -self.nodes[0].z + self.nodes[2].z
        dxdc = -self.nodes[0].x + self.nodes[3].x
        dydc = -self.nodes[0].y + self.nodes[3].y
        dzdc = -self.nodes[0].z + self.nodes[3].z

        matJ = np.array([[dxda, dyda, dzda],
                         [dxdb, dydb, dzdb],
                         [dxdc, dydc, dzdc]])        

        # ヤコビアンが負にならないかチェックする
        if LA.det(matJ) < 0:
            raise ValueError("要素の計算に失敗しました")

        return matJ

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    #---------------------------------------------------------------------
    def make_B_matrix(self):

        # dNdaの行列を計算する
        dNda = self.make_dNda()

         # ヤコビ行列を計算する
        matJ = self.make_J_matrix()

        #dNdxy = matJinv * matdNdab
        dNdxy = LA.solve(matJ, dNda)

        # Bマトリクスを計算する
        matB = np.array([[dNdxy[0, 0], 0.0, 0.0, dNdxy[0, 1], 0.0, 0.0, dNdxy[0, 2], 0.0, 0.0, dNdxy[0, 3], 0.0, 0.0],
                         [0.0, dNdxy[1, 0], 0.0, 0.0, dNdxy[1, 1], 0.0, 0.0, dNdxy[1, 2], 0.0, 0.0, dNdxy[1, 3], 0.0],
                         [0.0, 0.0, dNdxy[2, 0], 0.0, 0.0, dNdxy[2, 1], 0.0, 0.0, dNdxy[2, 2], 0.0, 0.0, dNdxy[2, 3]],
                         [0.0, dNdxy[2, 0], dNdxy[1, 0], 0.0, dNdxy[2, 1], dNdxy[1, 1], 0.0, dNdxy[2, 2], dNdxy[1, 2], 0.0, dNdxy[2, 3], dNdxy[1, 3]],
                         [dNdxy[2, 0], 0.0, dNdxy[0, 0], dNdxy[2, 1], 0.0, dNdxy[0, 1], dNdxy[2, 2], 0.0, dNdxy[0, 2], dNdxy[2, 3], 0.0, dNdxy[0, 3]],
                         [dNdxy[1, 0], dNdxy[0, 0], 0.0, dNdxy[1, 1], dNdxy[0, 1], 0.0, dNdxy[1, 2], dNdxy[0, 2], 0.0, dNdxy[1, 3], dNdxy[0, 3], 0.0]])

        return matB
    
    #---------------------------------------------------------------------
    # dNdxの行列を計算する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_dNda(self):
        
        # dNi/da, dNi/dbを計算する
        dN1da = -1.0
        dN2da = 1.0
        dN3da = 0.0
        dN4da = 0.0
        dN1db = -1.0
        dN2db = 0.0
        dN3db = 1.0
        dN4db = 0.0
        dN1dc = -1.0
        dN2dc = 0.0
        dN3dc = 0.0
        dN4dc = 1.0

        # dNi/dx, dNi/dyを計算する
        dNda = np.matrix([[dN1da, dN2da, dN3da, dN4da],
                          [dN1db, dN2db, dN3db, dN4db],
                          [dN1dc, dN2dc, dN3dc, dN4dc]])
        
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
            matB = self.make_B_matrix()
            
            # 構成則の内部変数の更新
            self.material[ip].compute_stress_and_tangent_matrix(matB, elem_solution)
    
    #---------------------------------------------------------------------
    # 構成則の変数を更新する
    #---------------------------------------------------------------------
    def update_constitutive_law(self):

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # 構成則の内部変数の更新
            self.material[ip].update()
