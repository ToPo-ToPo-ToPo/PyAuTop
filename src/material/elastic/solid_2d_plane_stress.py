# https://qiita.com/Altaka4128/items/eb4e9cb0bf46d450b03f

# kato

import numpy as np
import numpy.linalg as LA
from src.material.dmatrix_plane_stress import DmatrixPlaneStress

# =============================================================================
# 2次元ソリッド要素に対する等方性弾生体モデルの構成則を計算するためのクラス
# =============================================================================
class ElasticSolidPlaneStress:
    # コンストラクタ
    # young       : ヤング率
    # poisson     : ポアソン比
    # density     : 密度
    def __init__(self, young, poisson, density):
        # インスタンス変数を定義する
        self.young_list = [young, 1e-8]  # ヤング率のリスト[材料のヤング率, 設計変数を考慮したヤング率]
        self.density_list = [density, 1e-8]  # 密度のリスト[材料の質量密度, 設計変数を考慮した質量密度]

        self.poisson = poisson  # ポアソン比
        
        # 関連する変数を初期化する
        self.vecEStrain = np.zeros(3)  # 要素内の弾性ひずみ
        self.vecStress = np.zeros(3)  # 要素内の応力
        self.mises = 0.0  # 要素内のミーゼス応力
        self.z_strain = np.zeros(1) # z方向のひずみ
        self.epsilon = 1e-8
        
        # C0マトリックス（Dマトリックスのヤング率に依存しないところ）を作成する
        self.matC0 = DmatrixPlaneStress(self.young_list[-1], self.poisson).make_C0_matrix()
    
        # Dマトリックスを初期化する
        self.matD = self.young_list[-1] * self.matC0
        
    #---------------------------------------------------------------------
    # 材料物性を更新する
    #---------------------------------------------------------------------
    def update_parameter(self, design_density):
        self.young_list[-1] = (self.young_list[0] - self.epsilon) * design_density**3.0 + self.epsilon
        self.density_list[-1] = self.density_list[0] * design_density
        self.matD = self.young_list[-1] * self.matC0
                
    #---------------------------------------------------------------------
    # 応力を更新する
    # solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def compute_stress_and_tangent_matrix(self, matB, solution):
        
        self.young = self.young_list[-1] # 設計変数を考慮したヤング率
        self.density = self.density_list[-1] # 設計変数を考慮した質量密度
        
        # 全ひずみを求める
        vecStrain = matB @ solution 

        # 応力を求める
        self.vecStress = self.matD @ vecStrain

        # z方向のひずみを求める
        self.z_strain = - (self.poisson / self.young) * (self.vecStress[0] + self.vecStress[1])

        # mises応力を求める
        self.mises = self.mises_stress(self.vecStress)
    
    #---------------------------------------------------------------------
    # ニュートンラプソン法収束後の内部変数の更新
    #---------------------------------------------------------------------
    def update(self):
        pass

    #---------------------------------------------------------------------
    # ミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    #---------------------------------------------------------------------
    def mises_stress(self, vecStress):
        
        tmp1 = 0.5 * (vecStress[0] + vecStress[1])
        tmp2 = np.sqrt(np.square(0.5 * (vecStress[0]-vecStress[1])) + np.square(vecStress[2])) 
        max_p_stress = tmp1 + tmp2
        min_p_stress = tmp1 - tmp2
        mises = np.sqrt(0.5 * (np.square(max_p_stress-min_p_stress) + np.square(max_p_stress) + np.square(min_p_stress)))

        return mises

