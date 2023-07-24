
import numpy as np
import numpy.linalg as LA
from src.physics.element.C3D8 import C3D8
#=============================================================================
# 6面体8節点要素のクラス
#=============================================================================
class C3D8Bbar(C3D8):
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 要素(Node型のリスト)
    # material        : 構成則
    def __init__(self, no, nodes, material):

        # 継承元のC3D8クラスの__init__を使用する
        super().__init__(no, nodes, material)

    #---------------------------------------------------------------------
    # 要素接線剛性マトリクスKeを作成する
    #---------------------------------------------------------------------
    def make_K(self):

        # 初期化
        Ke = np.zeros([self.num_dof_at_node * self.num_node, self.num_dof_at_node * self.num_node])

        # 積分点ループ
        for ip in range(self.ipNum):

            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(ip)

            # Bbarマトリクスを計算する
            matBbar = self.make_Bbar_matrix(ip)

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
            matJ = self.make_J_matrix(ip)

            # Bbarマトリクスを計算する
            matBbar = self.make_Bbar_matrix(ip)

            # 内力ベクトルを計算する
            Fint_e += self.w1[ip] * self.w2[ip] * self.w3[ip] * matBbar.T @ self.material[ip].vecStress * LA.det(matJ)

        return Fint_e

    #---------------------------------------------------------------------
    # Bbarマトリクスを作成する
    # a : a座標値
    # b : b座標値
    # c : c座標値
    #---------------------------------------------------------------------
    def make_Bbar_matrix(self, ip):

        # Bマトリクスを作成する
        matB = self.make_B_matrix(ip)

        # Bvマトリクスを作成する
        matBv = self.make_Bv_matrix(ip)

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
    def make_Bv_matrix(self, ip):

         # dNdabcの行列を計算する
        matdNdabc = self.make_dNda(ip)

         # ヤコビ行列を計算する
        matJ = self.make_J_matrix(ip)

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

        # 初期化する
        Bvbar = np.zeros([6, self.num_node * self.num_dof_at_node])

        # 積分点ループ
        for ip in range(self.ipNum):
            
            # Bvマトリクスを計算する
            matBv = self.make_Bv_matrix(ip)

            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(ip)

            # ガウス積分でBvbarマトリクスを計算する
            Bvbar += self.w1[ip] * self.w2[ip] * self.w3[ip] * matBv * LA.det(matJ)
        
        # 体積を計算する
        v = self.get_volume()

        # 更新する
        Bvbar *= 1.0 / v

        return Bvbar

    #---------------------------------------------------------------------
    # 体積を求める
    #---------------------------------------------------------------------
    def get_volume(self):

        # 積分点ループ
        volume = 0
        for ip in range(self.ipNum):
            
            # ヤコビ行列を計算する
            matJ = self.make_J_matrix(ip)

            # ガウス積分で体積を計算する
            volume += self.w1[ip] * self.w2[ip] * self.w3[ip] * LA.det(matJ)

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
            matBbar = self.make_Bbar_matrix(ip)
            
            # 構成則の内部変数の更新
            self.material[ip].compute_stress_and_tangent_matrix(matBbar, elem_solution)