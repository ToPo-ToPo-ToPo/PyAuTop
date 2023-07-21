import numpy as np

#=============================================================================
# 1次元2節点トラス要素のクラス
#=============================================================================
class D1T2:
    # コンストラクタ
    # no         : 要素番号
    # nodes      : 節点のリスト(Node1d型のリスト)
    # material   : 材料特性(Material型)
    def __init__(self, no, nodes, material, area):

        # インスタンス変数を定義する
        self.no = no                  # 要素番号
        self.nodes = nodes            # 節点のリスト(Node1d型のリスト形式)
        self.material = []            # 材料モデルのリスト
        self.area = area              # 断面積
        
        self.num_node = 2             # 要素を構成する節点の数
        self.num_dof = 1              # 1節点が持つ自由度の数

        self.ipNum = 1                # 積分点の数
        self.w = 2.0                  # 積分点の重み係数

        # 積分点の数だけ材料モデルを初期化する
        for ip in range(self.ipNum):
            self.material.append(material)
    
    #---------------------------------------------------------------------
    # 要素接線剛性マトリクスKetを作成する
    #---------------------------------------------------------------------
    def make_local_Kt(self):

        # 初期化
        Ke = np.zeros([self.num_dof * self.num_node, self.num_dof * self.num_node])

        # 積分点ループ
        for ip in range(self.ipNum):
            # dxdaを計算する
            dxda = -0.5 * self.nodes[0].x + 0.5 * self.nodes[1].x
            
            # Bマトリクスを計算する
            matB = self.makeBmatrix(ip)

            # 要素接線剛性マトリクスKetを計算する
            Ke += self.area * self.w * matB.T * self.material[ip].D * matB * dxda

        return Ke

    #---------------------------------------------------------------------
    # 内力ベクトルFint_eを作成する
    #---------------------------------------------------------------------
    def make_local_Fint(self):

        # 初期化
        fe = np.zeros(self.num_dof * self.num_node)

        # 積分点ループ
        for ip in range(self.ipNum):
            # dxdaを計算する
            dxda = -0.5 * self.nodes[0].x + 0.5 * self.nodes[1].x
            
            # Bマトリクスを計算する
            matB = self.makeBmatrix(ip)

            # 内力ベクトルqを計算する
            fe += np.array(self.area * self.w * matB.T @ self.material[ip].stress * dxda).flatten()

        return fe
    
    #---------------------------------------------------------------------
    # 構成則の情報を更新する
    # elem_solution : 要素節点の変位ベクトル(np.array型)
    #---------------------------------------------------------------------
    def compute_constitutive_law(self, elem_solution):

        # 積分点ループ
        for ip in range(self.ipNum):

            # Bマトリクスを計算する
            matB = self.makeBmatrix(ip)

            # 全ひずみを求める
            strain = np.array(matB @ elem_solution).flatten()

            # リターンマッピング法により、応力、塑性ひずみ、降伏判定を求める
            self.material[ip].compute_stress_and_tangent_matrix(strain)

    #---------------------------------------------------------------------
    # Bマトリクスを作成する
    #---------------------------------------------------------------------
    def makeBmatrix(self, ip):

        # Bマトリクスの成分を計算する
        dN1dx = -1 / (-self.nodes[0].x + self.nodes[1].x)
        dN2dx = 1 / (-self.nodes[0].x + self.nodes[1].x)

        # 成分をマトリクスにまとめる
        matB = np.matrix([[dN1dx, dN2dx]])

        return matB
